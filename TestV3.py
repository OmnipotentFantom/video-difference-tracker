import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 打开视频文件
cap = cv2.VideoCapture(r"111.mp4")

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频帧大小
h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# 读取静态背景图像（没有人的图片）并调整大小
background_image = cv2.imread("background.jpg", cv2.IMREAD_GRAYSCALE)
if background_image is None:
    print("Error: Could not load background image.")
    exit()
background_image = cv2.resize(background_image, (w, h))

# 随机颜色数组，用于标记不同目标
max_corners = 200  # 初始的最大特征点数
color = np.random.randint(0, 255, (max_corners, 3))

# 初始化光流跟踪的特征点
prev_pts = None
prev_gray = None

# 创建一个空的轨迹图层
trajectory_layer = np.zeros((h, w, 3), dtype=np.uint8)

# 用于KNN分类器的特征数据和标签
knn_data = []
knn_labels = []
knn_model = KNeighborsClassifier(n_neighbors=3)

# 开始读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算当前帧与背景图像的差分
    diff_frame = cv2.absdiff(background_image, gray)

    # 二值化图像
    _, fgmask = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)

    # 去除噪声（形态学开操作）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # 提取轮廓用于标记移动目标
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []  # 存储所有有效的边界框

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # 过滤小面积的噪声
            continue
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    # 检测前景中的角点作为特征点
    if prev_pts is None:
        prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.3, minDistance=7, blockSize=7)
        prev_gray = gray

    # 如果有特征点，则进行光流计算
    if prev_pts is not None:
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

        if next_pts is not None:
            # 过滤出有效的点
            good_new = next_pts[status == 1]
            good_old = prev_pts[status == 1]

            # 准备KNN的数据和标签
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = int(new[0]), int(new[1])
                c, d = int(old[0]), int(old[1])

                # 将点的坐标作为特征，标签为框索引
                feature = [a, b, c, d]
                knn_data.append(feature)
                knn_labels.append(i % len(color))

                # 在轨迹图层上绘制运动轨迹
                cv2.line(trajectory_layer, (a, b), (c, d), color[i % len(color)].tolist(), 2)

            # 如果数据量足够，训练KNN模型
            if len(knn_data) > 50:
                knn_model.fit(knn_data, knn_labels)

            # 预测目标所属分类并绘制框
            for (x, y, w, h) in bounding_boxes:
                center_x, center_y = x + w // 2, y + h // 2
                prediction = knn_model.predict([[center_x, center_y, center_x - 5, center_y - 5]])
                label_color = color[prediction[0] % len(color)].tolist()
                cv2.rectangle(frame, (x, y), (x + w, y + h), label_color, 2)

            # 更新特征点和前一帧
            prev_pts = good_new.reshape(-1, 1, 2)
            prev_gray = gray
        else:
            prev_pts = None
    else:
        prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.3, minDistance=7, blockSize=7)

    # 将轨迹叠加到原图上
    combined_frame = cv2.addWeighted(frame, 0.8, trajectory_layer, 0.5, 0)

    # 显示结果
    cv2.imshow('Tracking with KNN', combined_frame)
    cv2.imshow('Foreground Mask', fgmask)

    # 按下 'q' 键退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()