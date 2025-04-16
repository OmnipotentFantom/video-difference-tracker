import cv2
import numpy as np

# 打开视频文件
cap = cv2.VideoCapture(r"11.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频帧大小和帧率
h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# 跳转到视频开头读取第一帧作为背景帧
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, background_frame = cap.read()
if not ret:
    print("Error: Could not read the background frame.")
    exit()

# 转换背景帧为灰度图并应用高斯模糊
background_image = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
background_image = cv2.GaussianBlur(background_image, (5, 5), 0)

# 使用全局直方图均衡化
background_image = cv2.equalizeHist(background_image)

# 定义感兴趣区域（ROI），支持动态选择
roi_mask = np.zeros((h, w), dtype=np.uint8)
cv2.rectangle(roi_mask, (0, 200), (w - 100, h), 255, -1)

# 定义形态学操作的核
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# 初始化拖影效果图层
trailing_effect = np.zeros((h, w), dtype=np.uint8)
trailing_decay = 255 // (frame_rate * 5)  # 拖影持续5秒

# 开始读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error.")
        break

    # 转换为灰度图并应用高斯模糊
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用全局直方图均衡化
    gray = cv2.equalizeHist(gray)

    # 计算当前帧与背景图像的绝对差
    diff_frame = cv2.absdiff(background_image, gray)

    # 二值化图像
    _, fgmask = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)

    # 应用感兴趣区域掩码
    fgmask = cv2.bitwise_and(fgmask, roi_mask)

    # 去除噪声（形态学开操作）
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)

    # 去除小孔（形态学闭操作）
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)

    # 更新拖影效果图层
    trailing_effect = cv2.addWeighted(trailing_effect, 1.0, fgmask, 1.0, 0)
    trailing_effect = cv2.subtract(trailing_effect, trailing_decay)

    # 将拖影效果叠加到原始帧
    trailing_overlay = cv2.cvtColor(trailing_effect, cv2.COLOR_GRAY2BGR)
    combined_frame = cv2.addWeighted(frame, 1.0, trailing_overlay, 0.5, 0)

    # 提取轮廓用于标记差异区域
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始帧上绘制多边形边界框
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:  # 面积过滤（根据需求调整）
            continue
        epsilon = 0.02 * cv2.arcLength(contour, True)  # 轮廓近似精度
        approx = cv2.approxPolyDP(contour, epsilon, True)  # 近似多边形
        cv2.drawContours(combined_frame, [approx], -1, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Difference Detection with Trailing Effect', combined_frame)
    cv2.imshow('Foreground Mask', fgmask)

    # 按下 'q' 键退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
