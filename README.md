# paper_scissors_rock
> 利用電腦視覺來進行手勢判斷
## 模型
利用 tensorflow 建構CNN模型，以及scikit-learn的分類器
## 訓練資料
利用 webcam 拍照，並自動做data label  
```python
while True:
    success, img = cap.read()
    if success:
        if img.size > 0:
            cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        # 根據鍵盤按鍵來把照片放入對應的資料夾
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
```
## 實作方法
由 multi thread 去執行不同鏡頭輸入影像去做手勢偵測，並成功分辨出 paper、scissors、rock 的手勢，  
並在輸出影像上加上剪刀、石頭、布的標示。  
![img](https://github.com/a0911078037/paper_scissors_rock/blob/main/result.png)
