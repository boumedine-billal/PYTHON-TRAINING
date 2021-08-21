from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate
from keras.models import load_model
import numpy as np
import time
import cv2


model_path = "C:\\Users\\moi\\Desktop\\formation_python_celec\\keras_models\\model07.h5"



# Build U-Net model
def unet(sz = (256, 256, 3)):
  x = Input(sz)
  inputs = x
  
  #down sampling 
  f = 4
  layers = []
  
  for i in range(0, 6):
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    layers.append(x)
    x = MaxPooling2D() (x)
    f = f*2
  ff2 = 64 
  
  #bottleneck 
  j = len(layers) - 1
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
  x = Concatenate(axis=3)([x, layers[j]])
  j = j -1 
  
  #upsampling 
  for i in range(0, 5):
    ff2 = ff2//2
    f = f // 2 
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j -1 
    
  
  #classification1
  x1 = Conv2D(f*2, 3, activation='relu', padding='same') (x)
  x1 = Conv2D(f, 3, activation='relu', padding='same') (x1)
  x1 = Conv2D(f, 3, activation='relu', padding='same') (x1)
  outputs1 = Conv2D(1, 1, activation='sigmoid') (x1)
  
  #classification2
  x2 = Conv2D(f*2, 3, activation='relu', padding='same') (x)
  x2 = Conv2D(f, 3, activation='relu', padding='same') (x2)
  x2 = Conv2D(f, 3, activation='relu', padding='same') (x2)
  outputs2 = Conv2D(1, 1, activation='sigmoid') (x2)
  
  #model creation 
  model = Model(inputs=[inputs], outputs=[outputs1,outputs2])
  model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
  model.summary()
  
  return model



model = unet()


model.save( "C:\\Users\\moi\\Desktop\\formation_python_celec\\keras_models\\model06.h5")

samples_images = 110


x_train  = np.zeros((samples_images, 256, 256, 3), dtype=np.float32)
y_train1 = np.zeros((samples_images, 256, 256, 1), dtype=np.float32)
y_train2 = np.zeros((samples_images, 256, 256, 1), dtype=np.float32)
 

tx  = "C:\\Users\\moi\\Desktop\\formation_python_celec\\scripts\\project2\\train\\x\\x"



ty1  = "C:\\Users\\moi\\Desktop\\formation_python_celec\\scripts\\project2\\train\\y1\\y"
ty2  = "C:\\Users\\moi\\Desktop\\formation_python_celec\\scripts\\project2\\train\\y2\\y"

p = 0

for i in range(samples_images):
    
    x = cv2.resize(cv2.imread(tx+str(i+1)+'.png'),(256,256))
    x_train[p,:,:,:] = x/255.0
    
    y1 = cv2.imread(ty1+str(i+1)+'.png')
    y_train1[p,:,:,0] = y1[:,:,0]/255.0
    
    y2 = cv2.imread(ty2+str(i+1)+'.png')
    y_train2[p,:,:,0] = y2[:,:,0]/255.0
    
    print(p)
    p += 1



# randomize data
index = np.arange(len(y_train1))
np.random.shuffle(index)
x_train = x_train[index,:,:,:]
y_train1 = y_train1[index,:,:,:]  
y_train2 = y_train2[index,:,:,:]  

results = model.fit(x_train, [y_train1, y_train2], validation_split=0.1, batch_size=4 ,epochs=60)#500 , validation_split=0.1
model.save(model_path)
























# new_model = load_model("mnist-model107.h5")
# results = new_model.fit(x_train, y_train, validation_split=0.1, batch_size=10 ,epochs=10)#50



"""
for i in range(12):
    x = np.zeros((1 , 256, 256, 3), dtype=np.uint8)
    x[0,:,:,:] = x_train[i,:,:,:]
    a10 = model.predict(x, verbose=1)[0,:,:,0]*255
    t1 = time.time()
    a1 = model.predict(x, verbose=1)[0,:,:,0]*255
    t2 = time.time()

    print(t2-t1)
    a3 = a1.astype(np.uint8)
    a4 = (x_train[i,:,:,:]).astype(np.uint8)

    cv2.imshow('frame1',a3)
    cv2.imshow('frame2',a4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""


















"""
#new_model = load_model("mnist-model107.h5")
new_model = model

x_train = np.zeros((20, 256, 256, 3), dtype=np.float)

 
for i in range(3):
    x = cv2.imread(tx2+str(i+1)+'.png')
    x_train[i,:,:,:] = x
    print(i+1)

import time
##########
for i in range(20):
    x = np.zeros((1 , 256, 256, 3), dtype=np.uint8)
    x[0,:,:,:] = x_train[i,:,:,:]
    a10 = new_model.predict(x, verbose=1)[0,:,:,0]*255
    t1 = time.time()
    a1 = new_model.predict(x, verbose=1)[0,:,:,0]*255
    t2 = time.time()

    print(t2-t1)
    a3 = a1.astype(np.uint8)
    a4 = (x_train[i,:,:,:]).astype(np.uint8)

    cv2.imshow('frame1',a3)
    cv2.imshow('frame2',a4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""    
