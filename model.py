import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

IMAGE_SIZE = (128,128)
IMAGE_FULL_SIZE = (128,128, 3)
batchSize = 8

allImages = np.load("temp\\allDogImages.npy")
allLabels = np.load("temp\\allDogLabels.npy")

print(allImages.shape)
print(allLabels.shape)

# convert the labels to integers
print(allLabels)

le = LabelEncoder()
integerLabels = le.fit_transform(allLabels)
print(integerLabels)

# unique integer labels
numOfCategories = len(np.unique(integerLabels)) # = 120
print(numOfCategories)

# convert the integer labels to categorical -> prepare for train
allLabelsForModel = to_categorical(integerLabels, num_classes=numOfCategories)
print(allLabelsForModel)

# normalize images
allImagesForModel = allImages.astype(np.float32) / 255.0

# create train and test data
X_train, X_test, y_train, y_test = train_test_split(allImagesForModel, allLabelsForModel, test_size=0.3)

print("X_train, X_test, y_train, y_test ----> shape:")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# free some memory
del allImages
del allLabels
del integerLabels
del allImagesForModel

# build moodel
myModel = NASNetLarge(input_shape=IMAGE_FULL_SIZE, weights='imagenet', include_top=False)

# we dont want to train the existing layer
for layer in myModel.layers:
    layer.trainable = False
    print(layer.name)

# add flatten layer
plusFlattenLayer = Flatten()(myModel.output)

# add the last dense layer without 120 classes
prediction = Dense(numOfCategories, activation='softmax')(plusFlattenLayer)

model = Model(inputs=myModel.input, outputs=prediction)

#print(model.summary())

lr = 1e-4 #0.0001
opt = Adam(lr)

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = opt,
    metrics = ['accuracy']
)

stepsPerEpoch = int(np.ceil(len(X_train) / batchSize))
validationSteps = int(np.ceil(len(X_test) / batchSize))

# early stopping
best_model_file = "temp\\dogs.keras"

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy', patience=7, verbose=1)
]

# train the model (fit)
r = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=batchSize,
    callbacks=[callbacks]
)