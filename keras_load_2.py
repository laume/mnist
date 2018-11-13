import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt
import PIL.ImageOps
from keras.preprocessing import image



def create_model():

    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = create_model()

# Load the weights with the best validation accuracy
model.load_weights('./models/model.weights.best.hdf5')


# Making new predictions

test_image = image.load_img('test_img/keturi.png', target_size = (28, 28))
test_image = image.img_to_array(test_image)
print(test_image.shape)
test_image = np.resize(test_image, (28, 28, 1))
#plt.imshow(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(result)
pred = model.predict_classes(test_image)[0]
print(pred)

## Checking with inverted image:

test_image = image.load_img('test_img/keturi.png', target_size = (28, 28))
plt.imshow(test_image)
inverted_image = PIL.ImageOps.invert(test_image)
plt.imshow(inverted_image)
test_image = image.img_to_array(inverted_image)
print(test_image.shape)
test_image = np.resize(test_image, (28, 28, 1))
#plt.imshow(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(result)
pred = model.predict_classes(test_image)[0]
print(pred)
