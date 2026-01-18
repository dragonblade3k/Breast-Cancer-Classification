# import necessary libraries
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
import os
import shutil
from uuid import uuid4
#from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from cancernet.cancernet import CancerNet
from tensorflow.keras.preprocessing import image

# load the saved model weights

model=CancerNet.build(width=48,height=48,depth=3,classes=2)

model.load_weights(r"E:\Major Project\Trained_Model.h5")

# compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# create an ImageDataGenerator object for data augmentation
test_data_gen = ImageDataGenerator(rescale=1./255)


# load the test dataset from the folders '0' and '1'
test_data = test_data_gen.flow_from_directory(r"E:\Major Project\idc\testing", target_size=(48, 48), batch_size=128)

# evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_data)

# print the model accuracy on the test dataset
print('Accuracy on test dataset:', accuracy)


# load an image to classify
# image_count = 100
# # cls_1_obj = []
# cls_0_obj = []

# PATH_1 = r"C:\Users\shant\OneDrive\Desktop\Major Project\breast-cancer-classification\breast-cancer-classification\datasets\idc\training\1"
# PATH_0 = r"C:\Users\shant\OneDrive\Desktop\Major Project\breast-cancer-classification\breast-cancer-classification\datasets\idc\training\0"
# print('---------------------- CLASS 1 ----------------------')
# for i, image_name in enumerate(os.listdir(PATH_1)):
#     img_path = os.path.join(PATH_1, image_name)
#     img = image.load_img(img_path, target_size=(48, 48))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = tf.keras.applications.resnet50.preprocess_input(x)
#     preds = model.predict(x)
#     print(preds)
#     # cls_1_images.append(x)
#     if (i == image_count):
#         break
# print('------------------------------------------------------------------')
# print()
# print('---------------------- CLASS 0 ----------------------')
# PATH_output_dir = PATH_0[:-1] + '_ouput'
# os.makedirs(PATH_output_dir, exist_ok=True)
# for i, image_name in enumerate(os.listdir(PATH_0)):
#     img_path = os.path.join(PATH_0, image_name)
#     output_folder = os.path.join(PATH_output_dir, os.path.basename(img_path).rsplit('.')[0], '0')
#     os.makedirs(output_folder)
#     output_img_path = os.path.join(output_folder, os.path.basename(img_path))
#     shutil.copyfile(img_path, output_img_path)
#     # img = image.load_img(img_path, target_size=(48, 48))
#     img = test_data_gen.flow_from_directory(output_folder, target_size=(48, 48), batch_size=128)
#     # x = image.img_to_array(img)
#     # x = np.expand_dims(x, axis=0)
#     # x = tf.keras.applications.resnet50.preprocess_input(x)
#     # cls_0_images.append(x)
#     preds = model.predict(img)
#     print(preds)
#     if (i == image_count):
#         break
# print('------------------------------------------------------------------')
# print()

# img = image.load_img(r"C:\Users\shant\OneDrive\Desktop\Major Project\breast-cancer-classification\breast-cancer-classification\datasets\idc\training\0\10308_idx5_x1001_y851_class0.png", target_size=(48, 48))
# img = test_data_gen.flow_from_directory(r"C:\Users\shant\OneDrive\Desktop\Major Project\breast-cancer-classification\breast-cancer-classification\datasets\idc\test\0", target_size=(48, 48), batch_size=128)
# trasnformed_img = test_data_gen
# preprocess the image
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = tf.keras.applications.resnet50.preprocess_input(x)

# predict the class of the image
# preds = model.predict(trasnformed_img)

# print(preds)
