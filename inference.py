#%%
import tensorflow as tf
import os, glob, random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

data_dir = 'data/test'
save_dir = "model/"

BATCH_SIZE = 8

IMG_HEIGHT = 224
IMG_WIDTH = 224

# The 1./255 is to convert from uint8 to float32 in ranget [0,1].
test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data_gen = test_image_generator.flow_from_directory(directory=data_dir,target_size=(224,224),batch_size=BATCH_SIZE,class_mode='categorical')

model = tf.keras.models.load_model(save_dir+f"01_weights.h5")
model.trainable=False
opt = tf.keras.optimizers.Adam(learning_rate=0.001) 
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
pred=model.predict(test_data_gen,batch_size=1,verbose=1,steps=len(test_data_gen))

results = model.evaluate(test_data_gen, batch_size=1)
print('test loss, test acc:', results)
#%%
bb = {v:k for k,v in test_data_gen.class_indices.items()}

rand_idx = np.random.choice(range(0,len(test_data_gen)), 8)


fig, m_axs = plt.subplots(4, 2, figsize = (4,12))
for (idx, c_ax) in zip(rand_idx, m_axs.flatten()):
        c_ax.imshow(test_data_gen[idx][0][0])
        c_ax.set_title('label: %s\nPredicted: %s' % (str(bb.get(test_data_gen[idx][1][0].argmax())),str(bb.get(pred[idx].argmax()))))
        c_ax.axis('off')
fig.savefig('trained_img_predictions.png', dpi = 300)

# %%
