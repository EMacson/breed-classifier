import numpy as np
import pandas as pd
import cv2
import os

IMAGE_SIZE = (128,128)
IMAGE_FULL_SIZE = (128,128, 3)

trainFolder = "data\\train"

# load csv
df = pd.read_csv("data\\labels.csv")
print("head of labels:")
print("===============")

print(df.head())
print(df.describe())

print("Group by labels: ")
groupLabels = df.groupby("breed")["id"].count()
print(groupLabels.head(10))

#display one image
imgPath = "data\\train\\0a0c223352985ec154fd604d7ddceabd.jpg"
img = cv2.imread(imgPath)
#cv2.imshow("img", img)
#cv2.waitKey(0)

# prepare all images and labels as a numpy array
allImages = []
allLabels = []

for ix, (image_name, breed) in enumerate(df[['id', 'breed']].values):
    img_dir = os.path.join(trainFolder, image_name + '.jpg')
    print(img_dir)

    img = cv2.imread(img_dir)
    resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    allImages.append(resized)
    allLabels.append(breed)

print(len(allImages))
print(len(allImages))

print("save the data:")
allImages = np.array(allImages, dtype=np.uint8)
allLabels = np.array(allLabels)
np.save("temp\\allDogImages.npy", allImages)
np.save("temp\\allDogLabels.npy", allLabels)


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(allLabels)

# Save label mapping for inference
class_names = label_encoder.classes_
np.save("temp\\allDogClassNames.npy", class_names)

# Use encoded labels (for model training)
allLabels = np.array(encoded_labels)

np.save("temp\\dogLabels.npy", allLabels)

print("finish save te data ...")