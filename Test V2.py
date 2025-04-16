import cv2
import numpy as np

# 打开视频文件
cap = cv2.VideoCapture(r"11.mp4")

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取第一帧
ret, frame1 = cap.read()
if not ret:
    print("Error: Could not read video frame.")
    cap.release()
    exit()

# 将第一帧转换为灰度图
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# 初始化特征点，改用更灵活的 maxCorners 参数以适应不同视频
max_corners = 200  # 增加特征点数，提高追踪效果
prev_pts = cv2.goodFeaturesToTrack(prvs, maxCorners=max_corners, qualityLevel=0.3, minDistance=7, blockSize=7)

# 随机颜色数组
color = np.random.randint(0, 255, (max_corners, 3))

# 开始读取视频帧并计算光流
while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break

    # 将当前帧转换为灰度图
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 使用光流金字塔Lucas-Kanade法追踪特征点
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prvs, next, prev_pts, None)

    # 过滤出成功匹配的点
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    # 遍历特征点并绘制特征点位置
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        a, b = int(a), int(b)

        # 绘制特征点
        cv2.circle(frame2, (a, b), 5, color[i].tolist(), -1)

    # 显示结果
    cv2.imshow('Moving Points', frame2)

    # 按下 'q' 键退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # 更新前一帧和特征点以便进行下一次迭代
    prvs = next
    prev_pts = good_new.reshape(-1, 1, 2)

# 释放视频捕获和销毁所有窗口
cap.release()
cv2.destroyAllWindows()
