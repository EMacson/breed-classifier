import tensorflow as tf
import os
import numpy as np
import cv2

modelFile = "temp\\dogs.keras"
model = tf.keras.models.load_model(modelFile)

#print(model.summary())

inputShape = (128,128)

allLabels = np.load("temp\\allDogLabels.npy")
categories = np.unique(allLabels)

# prepare image
def prepareImage(img):
    resized = cv2.resize(img, inputShape, interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized, axis=0)
    imgResult = imgResult / 255
    return imgResult

testImagePath = "testImage.jpeg"

#load image
img = cv2.imread(testImagePath)
imageForModel = prepareImage(img)

# prediction
resultArray = model.predict(imageForModel, verbose=1)
answers = np.argmax(resultArray, axis=1)

print(answers)

text = categories[answers[0]]
print(text)

font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img, text, (0,20), font, 1, (209,1977), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()