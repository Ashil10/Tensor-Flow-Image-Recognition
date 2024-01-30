import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import cv2
import numpy as np

# Load pre-trained EfficientNetB2 model
model = tf.keras.applications.EfficientNetB2(weights='imagenet')

# Load and preprocess the image
img_path = '/home/cs-ns-04/Downloads/apple.jpg'
img = image.load_img(img_path, target_size=(260, 260))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)
decoded_predictions = decode_predictions(predictions, top=1)[0][0]

# Display the results
print(f"Predicted class: {decoded_predictions[1]}, Probability: {decoded_predictions[2]}")
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import cv2
import numpy as np

# Load pre-trained EfficientNetB2 model
model = tf.keras.applications.EfficientNetB2(weights='imagenet')

# Load and preprocess the image
img_path = '/home/cs-ns-04/Downloads/apple.jpg'
img = image.load_img(img_path, target_size=(260, 260))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)
decoded_predictions = decode_predictions(predictions, top=1)[0][0]

# Display the results
print(f"Predicted class: {decoded_predictions[1]}, Probability: {decoded_predictions[2]}")
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import cv2
import numpy as np

# Load pre-trained EfficientNetB2 model
model = tf.keras.applications.EfficientNetB2(weights='imagenet')

# Load and preprocess the image
img_path = '/home/cs-ns-04/Downloads/apple.jpg'
img = image.load_img(img_path, target_size=(260, 260))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)
decoded_predictions = decode_predictions(predictions, top=1)[0][0]

# Display the results
print(f"Predicted class: {decoded_predictions[1]}, Probability: {decoded_predictions[2]}")
#https://ieeexplore.ieee.org/document/10021223
