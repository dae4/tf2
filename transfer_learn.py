#%%
import tensorflow as tf
import numpy as np
import tensorflow as tf
import random
import os
import shutil
import PIL

## gpu setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

## fix the seed
SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
EPOCHS=100
BATCH_SIZE = 16 * len(gpus)

data_dir = 'data'
save_dir = 'model'
train_dir=os.path.join(data_dir,"train")
val_dir =os.path.join(data_dir,"val")

#%%

## ready to data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

trainImage = ImageDataGenerator(rescale=1./255,horizontal_flip=True)
train_gen = trainImage.flow_from_directory(directory=train_dir,target_size=(224,224),batch_size=BATCH_SIZE)
validImage = ImageDataGenerator(rescale=1./255,horizontal_flip=False)
valid_gen = validImage.flow_from_directory(directory=val_dir,target_size=(224,224),batch_size=BATCH_SIZE)

# multi GPU 
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    base_model=MobileNetV2( weights='imagenet', input_tensor=None, input_shape=None,include_top = False, pooling='avg')
    base_model.trainable=True
    x = layers.Dense(len(valid_gen.class_indices), activation='softmax')(base_model.output)
    model = models.Model(base_model.input, x)
    print(model.summary())

    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])


#%%
from tensorflow.keras.callbacks import ModelCheckpoint
weight_path=os.path.join(save_dir,"{}_weights.h5".format('{epoch:02d}'))
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                            save_best_only=True, mode='min', save_weights_only = False)

callbacks_list = [checkpoint]
train_steps=len(train_gen)
val_steps=len(valid_gen)
model.fit(  
            train_gen,
            steps_per_epoch = train_steps,
            validation_steps = val_steps, 
            validation_data = valid_gen, 
            epochs = EPOCHS, 
            callbacks = callbacks_list,
            max_queue_size=64, workers=32
                                            )
