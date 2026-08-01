import tensorflow as tf
import tensorflowjs as tfjs
import json
import os

# Load model
model = tf.keras.models.load_model('model.h5')

# Simpan ke folder tujuan
output_dir = '../web/src/model'
os.makedirs(output_dir, exist_ok=True)
tfjs.converters.save_keras_model(model, output_dir)
print(f"Model TFjs disimpan di {output_dir}")

# Salin labels.json
with open('labels.json') as f:
    labels = json.load(f)
with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
    json.dump(labels, f)
print("Labels disalin.")