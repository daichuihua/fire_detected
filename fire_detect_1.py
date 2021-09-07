"""
自己创建模型并训练，用于火检
数据集1：https://www.kaggle.com/atulyakumar98/test-dataset
数据集2：https://www.kaggle.com/phylake1337/fire-dataset
"""
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

"""构建并训练模型"""
'''
TRAINING_DIR = "D:\\fire_detect\\Train"
training_datagen = ImageDataGenerator(rescale=1. / 255,
                                      horizontal_flip=True,
                                      rotation_range=30,
                                      height_shift_range=0.2,
                                      fill_mode='nearest')

VALIDATION_DIR = "D:\\fire_detect\\Validation"

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                       target_size=(224, 224),
                                                       class_mode='categorical',
                                                       batch_size=64)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=16)

from tensorflow.keras.optimizers import Adam
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(384, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc'])
history = model.fit(train_generator, steps_per_epoch=15, epochs=50, validation_data=validation_generator, validation_steps=15)

model.save("D:\\fire_detect\\fire_detected.h5")
'''

"""使用模型 """
new_model = tf.keras.models.load_model("D:\\fire_detect\\fire_detected.h5")
import numpy as np
# path ='D:\\web_spider\\fire\\0.jpg'
path ='D:\\fire_detect\\FIRE-SMOKE-DATASET\\Test\\Neutral\\image_30.jpg'
img = image.load_img(path, target_size=(224, 224)) #列表
x = image.img_to_array(img) #数组
x = np.expand_dims(x, axis=0) #向量
images = np.vstack([x])
classes = new_model.predict(images, batch_size=10)
print(classes)
print(classes[0][0])
if classes[0][0] >= 0.5:
    print("fire!fire!fire!help!")
else:
    print("It is ok,don't worry!")
