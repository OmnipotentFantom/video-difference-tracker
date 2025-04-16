import cv2
import numpy as np



cap = cv2.VideoCapture(r"11.mp4")
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
prev_pts = cv2.goodFeaturesToTrack(prvs, 100, 0.3, minDistance=7, blockSize=7)
color = np.random.randint(0,255,(100,3))
while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prvs, next, prev_pts, None)
    good_new = next_pts[status==1]
    good_old = prev_pts[status==1]
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)
        mask = cv2.line(hsv, (a,b),(c,d),color[i].tolist(),2)
        frame2 = cv2.circle(frame2,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame2, mask)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    prvs = next
    prev_pts = good_new.reshape(-1,1,2)