from cvzone.HandTrackingModule import HandDetector
import cv2
import joblib
import numpy as np

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 352)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
detector = HandDetector(detectionCon=0.7, maxHands=1)
model = joblib.load('./svc.joblib')
labels = ['paper', 'scissors', 'rock']
while True:
    success, img = cap.read()
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if success:
        hands = None
        if img.size > 0:
            hands = detector.findHands(img, draw=False)

        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList = hand1["lmList"]  # List of 21 Landmarks Points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1["center"]  # center of the hand cx,cy
            handType1 = hand1["type"]  # Hand Type Left orRight

            img = img[bbox1[1]:bbox1[1]+bbox1[3], bbox1[0]:bbox1[0]+bbox1[2]]
            if img.size > 0:
                img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_LINEAR)
                img = img / 255.0
                img_list = [img]
                img_list = np.array(img_list).flatten()
                predict = model.predict(img_list)
                pred = np.argmax(predict, axis=1)
                print(f"label:{labels[pred[0]]}")
        if img.size > 0:
            cv2.imshow("Image", img)
cap.release()
cv2.destroyAllWindows()
