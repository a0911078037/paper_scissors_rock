import os
import uuid
train_dir = './test_data'
labels = ['paper', 'rock', 'scissors']
for label in labels:
    folder = os.path.join(train_dir, label)
    for image in os.listdir(folder):
        os.rename(os.path.join(folder, image), os.path.join(folder, f"{str(uuid.uuid4())[:6]}.png"))