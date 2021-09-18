
import os
import numpy as np
import shutil

def data_split(data_dir):
    model =data_dir
    image=data_dir

    for cls in os.listdir(image):
        os.makedirs(str(model) +'/train/' + cls)
        os.makedirs(str(model) +'/val/' + cls)
        # Creating partitions of the data after shuffeling
        src = os.path.join(image, cls) # Folder to copy images from
        print(src)
        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, val_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)* (1 - 0.2))])
        train_FileNames = [str(src)+'/'+ name for name in train_FileNames.tolist()]
        val_FileNames = [str(src)+'/' + name for name in val_FileNames.tolist()]

        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Validation: ', len(val_FileNames))

        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, str(model) +'/train/' + cls)

        for name in val_FileNames:
            shutil.copy(name, str(model) +'/val/' + cls)


data_dir = 'data'
save_dir = 'model'
train_dir=os.path.join(data_dir,"train")
val_dir =os.path.join(data_dir,"val")

if not os.path.exists(train_dir):
    data_split(data_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
