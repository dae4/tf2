from tensorflow import keras
model = keras.models.load_model('C:/Users/3210m/Desktop/project/tf2/model/01_weights.h5', compile=False)

export_path = 'tflite'
model.save(export_path, save_format="tf")