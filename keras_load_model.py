import numpy as np
from keras.models import load_model

img_width, img_height = 28, 28
num_classes = 10
input_shape = (img_width, img_height, 1)

my_model_pth = './models/keras_model.h5'
my_model = load_model(my_model_pth)

from keras.preprocessing import image
test_image = image.load_img('test_img/sesi.png', target_size = (28, 28))
test_image = image.img_to_array(test_image)
print(test_image.shape)
test_image = np.resize(test_image, (img_width, img_height, 1))
test_image = np.expand_dims(test_image, axis = 0)
result = my_model.predict(test_image)
print(result)
pred = my_model.predict_classes(test_image)[0]
print(pred)