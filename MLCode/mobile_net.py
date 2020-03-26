from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
data_path = pathlib.Path('../IMAGE')

all_image_paths = list(data_path.glob('*/*/*.png'))
all_image_paths = [str(path) for path in all_image_paths]
image_count = len(all_image_paths)
random.shuffle(all_image_paths)

label_names = sorted(
    item.name for item in data_path.glob('./*/*') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(
    path).parent.name] for path in all_image_paths]


def perprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0
    return image


all_images = [perprocess_image(x) for x in all_image_paths]
print(type(all_images))
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
print(path_ds)
image_ds = path_ds.map(
    perprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE,)
print(image_ds)
label_ds = tf.data.Dataset.from_tensor_slices(
    tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
print(image_label_ds)

BATCH_SIZE = 32

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)

ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

mobile_net = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False)
mobile_net.trainable = False


def change_range(image, label):
    return 2*image-1, label


keras_ds = ds.map(change_range)
image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names), activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

print(len(model.trainable_variables))
print(model.summary())

steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()

print("\n\nTraining results")
print("--------------------------------------------------------------------------------------")

model.fit(ds, epochs=10, steps_per_epoch=8)
print("\n\n")