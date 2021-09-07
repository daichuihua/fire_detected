# 数据集3：https://github.com/DeepQuestAI/Fire-Smoke-Dataset
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.optimizers import SGD

TRAINING_DIR = "D:\\fire_detect\\FIRE-SMOKE-DATASET\\Train"
training_datagen = ImageDataGenerator(rescale=1./255,
                                      zoom_range=0.15,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

VALIDATION_DIR = "D:\\fire_detect\\FIRE-SMOKE-DATASET\\Test"
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                       target_size=(224, 224),
                                                       shuffle=True,
                                                       class_mode='categorical',
                                                       batch_size=128)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size=(224, 224),
                                                              class_mode='categorical',
                                                              shuffle=True,
                                                              batch_size=14)

input_tensor = Input(shape=(224, 224, 3))
base_model = InceptionV3(input_tensor=input_tensor,
                         weights='imagenet',
                         include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(train_generator, steps_per_epoch=14, epochs=20, validation_data=validation_generator, validation_steps=14)

model.save("D:\\fire_detect\\fire_detected_InceptionV3.h5")

"""
#To train the top 2 inception blocks, freeze the first 249 layers and unfreeze the rest.for layer in model.layers[:249]:
layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True #Recompile the model for these modifications to take effectfrom
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(train_generator, steps_per_epoch=14, epochs=10, validation_data=validation_generator, validation_steps=14)
    
model.save("D:\\fire_detect\\fire_detected_InceptionV3_sec.h5")
"""
