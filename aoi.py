import pandas as pd
import numpy as np
from sklearn.svm import SVC
import cv2
import joblib
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score
import matplotlib.pyplot as plt


def test(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

if __name__ == '__main__':
    file_path = './data/annotations/train.json'
    df = pd.read_json(file_path, orient='index').transpose()
    images_df = df['images'].dropna()
    category_df = df['categories'].dropna()
    annotations_df = df['annotations'].dropna()

    category_list = [{'id': i['id'], 'name': i['name']} for i in category_df]

    input_path = './data/train'
    train_data = []
    train_labels = []

    for img_file in images_df:
        for row in annotations_df:
            if img_file['id'] == row['image_id']:
                img = cv2.imread(f'{input_path}/{img_file["file_name"]}', 0)
                img = cv2.resize(img, (640, 480))
                bbox_list = row['bbox']
                bbox_list = [int(x) for x in bbox_list]
                img = img[bbox_list[1]:bbox_list[1] + bbox_list[3], bbox_list[0]:bbox_list[0] + bbox_list[2]]

                img = cv2.resize(img, (64, 128))
                img = np.dstack([img, img, img])
                img = img.astype('float32') / 255
                train_data.append(img.flatten())
                train_labels.append(row['category_id'])

    file_path = './data/annotations/test.json'
    df = pd.read_json(file_path, orient='index').transpose()
    images_df = df['images'].dropna()
    category_df = df['categories'].dropna()
    annotations_df = df['annotations'].dropna()

    category_list = [{'id': i['id'], 'name': i['name']} for i in category_df]
    input_path = './data/test'
    test_data = []
    test_labels = []

    for img_file in images_df:
        for row in annotations_df:
            if img_file['id'] == row['image_id']:
                img = cv2.imread(f'{input_path}/{img_file["file_name"]}', 0)
                img = cv2.resize(img, (640, 480))
                bbox_list = row['bbox']
                bbox_list = [int(x) for x in bbox_list]

                img = img[bbox_list[1]:bbox_list[1] + bbox_list[3], bbox_list[0]:bbox_list[0] + bbox_list[2]]
                img = cv2.resize(img, (64, 128))
                img = np.dstack([img, img, img])
                img = img.astype('float32') / 255
                test_data.append(img.flatten())
                test_labels.append(row['category_id'])

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    svm_model = SVC(gamma='auto', kernel='linear')
    svm_model.fit(train_data, train_labels)

    # joblib.dump(svm_model, 'model/svm_linear.joblib')
    # svm_model = joblib.load('./model/svm_linear.joblib')

    # cross validation
    # result = cross_val_score(svm_model, train_data, train_labels, cv=5, scoring='accuracy')
    # print(result)
    # print(result.mean())

    # predicting test dataset
    error = 0
    for test, answer in zip(test_data, test_labels):
        result = svm_model.predict(test.reshape(1, -1))
        print(f"predict: {result[0]}, answer: {answer}")
        if result[0] != answer:
            error += 1

    print(f"total test_data: {len(test_labels)}")
    print(f"error:{error}")

    # Confusion Matrix
    # result = cross_val_predict(svm_model, train_data, train_labels, cv=5)
    # confusion_matrix = confusion_matrix(train_labels, result)
    # print(confusion_matrix)

    # f1 score
    # result = cross_val_predict(svm_model, train_data, train_labels, cv=5)
    # f1_score = f1_score(train_labels, result)
    # print(f1_score)

    # Precision versus recall
    # result = cross_val_predict(svm_model, train_data, train_labels, cv=5)
    # precisions, recalls, thresholds = precision_recall_curve(train_labels, result)
    #
    # recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
    # threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
    #
    # plt.figure(figsize=(8, 6))
    # test(precisions, recalls)
    # plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
    # plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
    # plt.plot([recall_90_precision], [0.9], "ro")
    # plt.show()
