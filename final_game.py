import keyboard
from cvzone.HandTrackingModule import HandDetector
from PIL import Image, ImageDraw, ImageFont
import cv2
import tensorflow as tf
import numpy as np
import threading

detector = HandDetector(detectionCon=0.6, maxHands=1)
model = tf.keras.models.load_model('model_cnn_final_4.h5')
labels = ['布', '剪刀', '石頭']


class camThread(threading.Thread):
    def __init__(self, previewName, camID, stop_event, game_event):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.stop_event = stop_event
        self.game_event = game_event

    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID, self.stop_event, self.game_event)


def camPreview(previewName, camID, stop_event, game_event):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 352)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
    fontpath = 'NotoSansTC-Regular.otf'
    font = ImageFont.truetype(fontpath, 25)

    while True:
        success, img = cam.read()
        if success and img.size > 0:
            if game_event.is_set():
                hands = detector.findHands(img, draw=False)
                if hands:
                    # hand1 = hands[0]
                    # lmList = hand1["lmList"]  # List of 21 Landmarks Points
                    # bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
                    # centerPoint1 = hand1["center"]  # center of the hand cx,cy
                    # handType1 = hand1["type"]  # Hand Type Left orRight

                    a = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
                    a = a / 255.0
                    img_list = [a]
                    img_list = np.array(img_list)
                    img_list = tf.data.Dataset.from_tensors(img_list)
                    predict = model.predict(x=img_list, verbose=0)
                    pred = np.argmax(predict, axis=1)
                    print(f"[{previewName}]label:{labels[pred[0]]}")

                    text = f"{labels[pred[0]]}"
                    imgPil = Image.fromarray(img)
                    draw = ImageDraw.Draw(imgPil)
                    draw.text((75, 0), text, fill=(0, 0, 0), font=font, align='center')
                    img = np.array(imgPil)
                else:
                    print(f"{previewName} hands not found")
            if stop_event.is_set():
                break

            cv2.imshow(previewName, img)
            k = cv2.waitKey(20)

        if stop_event.is_set():
            break

    cv2.destroyWindow(previewName)
    cam.release()


if __name__ == "__main__":
    stop_event = threading.Event()
    game_event = threading.Event()
    thread1 = camThread("Cam1", 2, stop_event, game_event)
    thread2 = camThread("Cam2", 1, stop_event, game_event)
    thread1.start()
    thread2.start()
    while True:
        key = keyboard.read_key()
        if key == "q":
            stop_event.set()
            thread1.join()
            thread2.join()
            break
        elif key == 's':
            game_event.set()
