import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import skimage
from skimage import transform
from skimage.color import rgb2gray
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import MaxPooling2D
# model visualization
from keras.utils import plot_model
# transfer learning pre-trained model
from keras.applications.resnet50 import ResNet50

config = tf.ConfigProto(log_device_placement=True)
config = tf.ConfigProto(allow_soft_placement=True)

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".jpg")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    # print(images)
    return images, labels

### parameters ###

ROOT_PATH = r"path\to\dataset" ### 1.define path to dataset root path

NUM_CLASSES = 5 ### 2.change class number

WEIGHTS_FILE_NAME = 'flowers.h5' ### 3.set weights file name

input_shape = (224, 224, 3)

### 4.tune parameters
dropout_parameter = 0.25
learning_rate = 0.1
epochs = 25
batch_size = 8
### parameters ###

# use folder name as label in Training and Testing
train_data_directory = os.path.join(ROOT_PATH, r"flowers\Training")
test_data_directory = os.path.join(ROOT_PATH, r"flowers\Testing")

# data pre process
# ----------------------------TRAIN DATA--------------------------------
# load data
# load training data into "x_train_images" and "y_train_labels" variable
x_train_images, y_train_labels = load_data(train_data_directory)

# images pre-processing resize -> array -> grayscale
# resize image to 224 x 224
x_train_images_224 = [transform.resize(
    image, (224, 224)) for image in x_train_images]

# Convert `x_train_images_224` to grayscale
# x_train_images_224 = rgb2gray(x_train_images_224)

print(type(x_train_images_224))
# plt.imshow(x_train_images_224[0])
# plt.show()

# Convert `x_train_images_224` to an np.array
x_train_images_224 = np.asarray(x_train_images_224)
print(type(x_train_images_224))
print(x_train_images_224.shape)

# print(type(y_train_labels))
y_train_labels = np.asarray(y_train_labels)
# print(y_train_labels.shape)
# print(y_train_labels)

y_train_labels = keras.utils.to_categorical(
    y_train_labels, num_classes=5, dtype='float32')
# print(y_train_labels)
# print(y_train_labels.shape)
# ----------------------------TRAIN DATA--------------------------------

# ----------------------------TEST DATA--------------------------------
# load testing data into "x_train_images" and "y_train_labels" variable
x_test_images, y_test_labels = load_data(test_data_directory)

# images pre-processing resize -> array -> grayscale
# resize image to 224 x 224
x_test_images_224 = [transform.resize(
    image, (224, 224)) for image in x_test_images]
print(type(x_test_images_224))
# Convert `x_test_images_224` to an np.array
x_test_images_224 = np.asarray(x_test_images_224)
print(type(x_test_images_224))
print(x_test_images_224.shape)

y_test_labels = np.asarray(y_test_labels)

y_test_labels = keras.utils.to_categorical(
    y_test_labels, num_classes=NUM_CLASSES, dtype='float32')
# ----------------------------TEST DATA--------------------------------

# The core data structure of Keras is a model(a way to organize layers). Sequential() is the simplest type of model.
# create VGG16 model
"""
model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same', data_format="channels_last",
           activation='relu'),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(128, (2, 2), strides=(2, 2), activation='relu', padding='valid'),
    # MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
    Dropout(dropout_parameter),

    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(256, (2, 2), strides=(2, 2), activation='relu', padding='valid'),
    # MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
    Dropout(dropout_parameter),

    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(512, (2, 2), strides=(2, 2), activation='relu', padding='valid'),
    # MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(dropout_parameter),

    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(512, (2, 2), strides=(2, 2), activation='relu', padding='valid'),
    # MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(dropout_parameter),

    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
    Conv2D(512, (2, 2), strides=(2, 2), activation='relu', padding='valid'),
    # MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(dropout_parameter),

    GlobalAveragePooling2D(data_format='channels_last'),
    # Flatten(),
    # Dense(4096, activation='relu'),
    # Dense(4096, activation='relu'),
    Dense(5, activation='softmax')
])

model.summary()
"""

# transfer learning
model = ResNet50(
    include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling=None, classes=NUM_CLASSES) ### 2.change class number
# configure its learning process
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=['accuracy']) # epoch 20 acc=0.66
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0), metrics=['accuracy'])


# x_train_images and y_train_labels are Numpy arrays
# model.fit(x_train_images_224, y_train_labels, epochs=5, batch_size=32)
history = model.fit(x_train_images_224, y_train_labels, validation_data=(x_test_images_224, y_test_labels),
                    epochs=epochs, batch_size=batch_size, verbose=1)
# , validation_split=0.25

model.save(WEIGHTS_FILE_NAME) ### 3.set weights file name

# Evaluate model performance
# loss_and_metrics = model.evaluate(x_test_images, y_test_labels, batch_size=128)

# generate predictions on new data
# classes = model.predict(x_test, batch_size=128)

# model visualization
# plot_model(model, to_file='model.png')

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
