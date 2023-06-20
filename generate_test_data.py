import cv2
import os

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

if not os.path.exists('./test_data'):
    os.mkdir('./test_data')

if not os.path.exists('./test_data/paper'):
    os.mkdir('./test_data/paper')

if not os.path.exists('./test_data/scissors'):
    os.mkdir('./test_data/scissors')

if not os.path.exists('./test_data/rock'):
    os.mkdir('./test_data/rock')

img_count = {
    "s": 0, "r": 0, "p": 0
}

while True:
    success, img = cap.read()
    if success:
        if img.size > 0:
            cv2.imshow("Image", img)

        key = cv2.waitKey(1)

        if key == ord('s'):
            cv2.imwrite(f'./test_data/scissors/{img_count["s"]}.png', img)
            img_count['s'] += 1
            print(f"剪刀已經拍:{img_count['s']+1}張")

        if key == ord('r'):
            cv2.imwrite(f'./test_data/rock/{img_count["r"]}.png', img)
            img_count['r'] += 1
            print(f"石頭已經拍:{img_count['r'] + 1}張")

        if key == ord('p'):
            cv2.imwrite(f'./test_data/paper/{img_count["p"]}.png', img)
            img_count['p'] += 1
            print(f"布已經拍:{img_count['p'] + 1}張")

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
