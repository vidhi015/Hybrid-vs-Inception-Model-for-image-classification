!pip install gradio

import gradio as gr

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

roses = list(data_dir.glob('roses/*'))
print(roses[5])
PIL.Image.open(str(roses[5]))

img_height,img_width=180,180
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

num_classes = 5

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes,activation='softmax')
]

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  train_ds,
  steps_per_epoch=8,
  validation_data=val_ds,
  epochs=epochs

epochs= 10
from tensorflow.keras.applications import InceptionV3
model1= Sequential()

pretrained_model= InceptionV3(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',classes=5,
                   weights='imagenet')
print(len(pretrained_model.layers))

for layer in pretrained_model.layers:
        layer.trainable=False
model1.add(layers.experimental.preprocessing.Rescaling(1./255,input_shape=(180,180,3)))
model1.add(pretrained_model)
model1.add(layers.Flatten())
model1.add(layers.Dense(256, activation='relu'))
model1.add(layers.Dense(128, activation='relu'))
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(num_classes,activation='softmax'))

model1.summary()

model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model1.save('C:/Users/KESHW/Downloads')

model1.fit(
  train_ds,
  validation_data=val_ds,
  steps_per_epoch=5,
  epochs=epochs
)

from PIL import Image
import cv2
def classify_with_hybrid(im):
  print('iMAGE1',im)
  im = Image.fromarray(im.astype('uint8'), 'RGB')
  im = im.resize((180,180))
  arr = np.array(im).reshape((-1, 180,180, 3))
  prediction=model.predict(arr)[0]
  return {class_names[i]: float(prediction[i]) for i in range(5)}

def classify_with_inception(im):
  print('iMAGE 2',im)
  im = Image.fromarray(im.astype('uint8'), 'RGB')
  im = im.resize((180,180))
  arr = np.array(im).reshape((-1,180,180, 3))
  arr = tf.keras.applications.inception_v3.preprocess_input(arr)
  prediction = model1.predict(arr).flatten()
  return {class_names[i]: float(prediction[i]) for i in range (5)}

import requests
from urllib.request import urlretrieve
urlretrieve("https://www.thephotoargus.com/wp-content/uploads/2020/02/rosepic12.jpg","rose.jpg")
urlretrieve("https://media.istockphoto.com/vectors/sunflower-flower-isolated-vector-id927047528?k=6&m=927047528&s=612x612&w=0&h=KZZ734lZ6zsEtw7zwi9QaSIsIXaQ9Us-mLTZ-AFSTPA=","sunflower.jpg")
urlretrieve("https://upload.wikimedia.org/wikipedia/commons/c/ce/Daisy_G%C3%A4nsebl%C3%BCmchen_Bellis_perennis_01.jpg","daisy.jpg");

print(keras.__version__)

imagein = gr.inputs.Image()
label = gr.outputs.Label(num_top_classes=3)

sample_images = [
                 ["rose.jpg"],
                 ["sunflower.jpg"],
                 ["daisy.jpg"]
]

gr.Interface(
    [classify_with_hybrid,classify_with_inception],
    imagein,
    label,
    title="Hybrid Model vs. InceptionNet",
    description="Our hybrid model gives faster and accurate results than the Inception model",
    examples=sample_images
).launch(debug='True');