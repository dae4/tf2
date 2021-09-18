#%%
import time
import tensorflow as tf
import numpy as np
def TFLiteInference(model_path,x_test,y_test):

    #Step 1. Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get indexes of input and output layers
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    sum_correct=0.0
    sum_time=0.0
    for idx, data in enumerate(zip(x_test,y_test)):
        image=data[0]
        label=data[1]
        image=tf.expand_dims(image, axis=0) #shape will be [1,32,32,3]
        
        s_time=time.time()
        #Step 2. Transform input data
        interpreter.set_tensor(input_index,image)
        #Step 3. Run inference
        interpreter.invoke()
        #Step 4. Interpret output
        pred=interpreter.get_tensor(output_index)
        
        sum_time+=time.time()-s_time
        if np.argmax(pred)== np.argmax(label):
            sum_correct+=1.0
    
    mean_acc=sum_correct / float(idx+1)
    mean_time=sum_time / float(idx+1)

    print(f'Accuracy of TFLite model: {mean_acc}')
    print(f'Inference time of TFLite model: {mean_time}')



#%%
interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()
# %%
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# %%

print("in",input_details)
print("out",output_details)
# %%
from PIL import Image
path = 'data/val/none/ILSVRC2012_val_00000080.JPEG'
load_img_rz = np.array(Image.open(path).resize((224,224)))
input_data = np.array([load_img_rz],dtype=np.float32)
input_data = input_data/255
# %%
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
# %%
output_data = interpreter.get_tensor(output_details[0]['index'])
print(np.argmax(output_data, axis=1))
# %%
