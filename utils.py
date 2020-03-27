import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import json
import h5py
from PIL import Image
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_model(model_path):
	reloaded_keras_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
	return reloaded_keras_model

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image /= 225
    image = tf.image.resize(image, (224, 224))
    image = image.numpy()
    return image

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    expanded_image = np.expand_dims(processed_image, axis=0)
    a = model.predict(expanded_image).squeeze()
    classes = np.argsort(a)[-top_k:]
    probs = [a[i] for i in classes]
    return probs, classes

def show_prediction(image_path, model, top_k, class_names_path):
	with open(class_names_path, 'r') as f:
		class_names = json.load(f)

	probs, classes = predict(image_path, model, top_k)
	categories = [class_names[str(int(i+1))] for i in classes]

	im = Image.open(image_path)
	test_image = np.asarray(im)
	processed_test_image = process_image(test_image)

	fig, (ax1, ax2) = plt.subplots(figsize=(10,10), nrows=2, ncols=1)
	ax1.imshow(processed_test_image)
	ax1.set_title(image_path)
	ax2.barh(categories, probs, height=0.3, align='center')
	ax2.set_yticks(categories)

	ax2.invert_yaxis()
	plt.tight_layout()
	plt.show()