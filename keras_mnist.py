import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.datasets import mnist
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


classifier = Sequential()

classifier.add(Conv2D(32, kernel_size=(3, 3),
                      input_shape = input_shape,
                      activation = 'relu'))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(num_classes, activation = 'softmax'))

classifier.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.Adadelta(),
                   metrics = ['accuracy'])


classifier.fit(x_train, y_train,
               batch_size=batch_size,
               epochs = epochs,
               verbose=1,
               validation_data = (x_test, y_test))


score = classifier.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])


# save the model
my_model = './models/keras_model.h5'
classifier.save(my_model)

# new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/test_img/sesi.png', target_size = (28, 28))
test_image = image.img_to_array(test_image)
print(test_image.shape)
test_image = np.resize(test_image, (img_rows, img_cols, 1))
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
pred = classifier.predict_classes(test_image)[0]
print(pred)
