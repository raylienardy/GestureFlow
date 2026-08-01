from tensorflow import keras
import tensorflowjs as tfjs

model = keras.models.load_model("../trainer/model.h5", compile=False)

tfjs.converters.save_keras_model(
    model,
    "../web/src/model"
)

print("Konversi selesai! File ada di web/src/model/")