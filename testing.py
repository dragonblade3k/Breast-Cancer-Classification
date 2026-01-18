# from keras.models import load_weights
import numpy as np
import tensorflow as tf
from cancernet.cancernet import CancerNet
from keras.layers.serialization import image_preprocessing
from tensorflow.python import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img,img_to_array
import keras


model=CancerNet.build(width=48,height=48,depth=3,classes=2)

#model = load_weights(r"C:\Users\shant\OneDrive\Desktop\Major Project\breast-cancer-classification\breast-cancer-classification\Trained_Model.h5")

model.load_weights(r"C:\Users\shant\OneDrive\Desktop\Major Project\breast-cancer-classification\breast-cancer-classification\Trained_Model.h5")
model.compile(optimizer='adam', loss = keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

def preprocessingImages1(path):
  image_data = ImageDataGenerator(zoom_range= 0.2, shear_range= 0.2, rescale= 1/225, horizontal_flip= True)
  image = image_data.flow_from_directory(directory= path, target_size = (224,224), batch_size = 32, class_mode = 'binary')
  
  return image

# path = "datasets\idc\testing"
# train_data = preprocessingImages1(path)


def preprocessingImages2(path):
  image_data = ImageDataGenerator(rescale= 1/225)
  image = image_data.flow_from_directory(directory= path, target_size = (224,224), batch_size = 32, class_mode = 'binary')
  
  return image

path = r"C:\Users\shant\OneDrive\Desktop\Major Project\breast-cancer-classification\breast-cancer-classification\datasets\idc\test"
test_data = preprocessingImages2(path)

# path = "/content/val"
# val_data = preprocessingImages2(path)


acc = model.evaluate(test_data)[1]

print(f"accuracy of model is {acc*100} %")

path = r"C:\Users\shant\OneDrive\Desktop\Major Project\breast-cancer-classification\breast-cancer-classification\datasets\idc\test"

img = load_img(path, target_size = (224,224))
input_arr =img_to_array(img)/225

plt.imshow(input_arr)
plt.show()

input_arr.shape

input_arr = np.expand_dims(input_arr, axis = 0)

pred = np.argmax(model.predict(input_arr))


if pred == 0:
  print("not a cancer")
else :
  print("a cancer")

# train_data.class_indices