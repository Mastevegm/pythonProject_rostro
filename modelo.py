import numpy
import datetime
import keras
import tensorflow as tf
import re
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Activation, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from os import listdir
from os.path import isfile, isdir, join


ih, iw = 192, 192 #Imagen alto y ancho
input_shape = (ih, iw, 3) # Se agrega número de canales

train_dir = 'C:/Users/train/' 
test_dir = 'C:/Users/test/'  #

num_class = 2 
epochs = 10 # Mientras mayor tengamos epoch, mejor será la precisión
batch_size = 50 
num_train = 1000 #numero de imagenes en train
num_test = 200 #numero de imagenes en test

epoch_steps = num_train // batch_size
test_steps = num_test // batch_size


gentrain = ImageDataGenerator(rescale=1. / 255.) 
train = gentrain.flow_from_directory(train_dir,
                #batch_size
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')
gentest = ImageDataGenerator(rescale=1. / 255)
test = gentest.flow_from_directory(test_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')

nuevo_model = tf.keras.models.load_model('test.h5')


model = keras.models.Sequential()
model.add(nuevo_model)
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='softmax'))
for layer in model.layers[:1]:
    layer.trainable = False

#Llamamos a tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
model.summary()


model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()],
              run_eagerly=True)

model.fit_generator
                train,
                steps_per_epoch=epoch_steps,
                epochs=epochs,
                validation_data=test,
                validation_steps=test_steps,
                callbacks=[tbCallBack]
                )

model.save('ModeloFace')
