from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate
from keras.models import load_model
import numpy as np
import time
import cv2

"""


tx  = "C:\\Users\\moi\\Desktop\\formation_python_celec\\scripts\\train\\x\\x"
ty  = "C:\\Users\\moi\\Desktop\\formation_python_celec\\scripts\\train\\y\\y"

model_path = "C:\\Users\\moi\\Desktop\\formation_python_celec\\keras_models\\model02.h5"

samples_images = 50



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
    
  
  #classification 
  x = Conv2D(f*2, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  outputs = Conv2D(1, 1, activation='sigmoid') (x)
  
  #model creation 
  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
  model.summary()
  
  return model



model = unet()





x_train = np.zeros((samples_images, 256, 256, 3), dtype=np.float)
y_train = np.zeros((samples_images, 256, 256, 1), dtype=np.float)

 
for i in range(samples_images):
    x = cv2.resize(cv2.imread(tx+str(i+1)+'.png'),(256,256))
    y = cv2.imread(ty+str(i+1)+'.png')
    x_train[i,:,:,:] = x/256.0
    y_train[i,:,:,0] = y[:,:,0]/255.0
    print(i+1)




results = model.fit(x_train, y_train, validation_split=0.1, batch_size=1 ,epochs=25)#500 , validation_split=0.1
model.save(model_path)





"""


















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
