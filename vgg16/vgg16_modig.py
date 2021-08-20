
# save the final model to file
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
import os
import numpy as np
import cv2




path = 'C:\\Users\\moi\\Desktop\\formation_python_celec\\vgg_classification\\fruits'






dirs = os.listdir(path)
classes = len(dirs)
dirs_main = []
size = 0
for i in range(classes):
    images_dirs = os.listdir(path+'\\'+dirs[i])
    len_images = len(images_dirs)
    dirs_im = []
    for j in range(len_images):
        image_path = path+'\\'+dirs[i]+'\\'+images_dirs[j]
        dirs_im.append(image_path)
        size += 1
    dirs_main.append(dirs_im)

x_train = np.zeros((size, 224, 224, 3), dtype=np.float32)
y_train = np.zeros((size, classes), dtype=np.float32)
    
p = 0
for i in range(classes):
    dirs_im = dirs_main[i]
    for j in range(len(dirs_im)):
        x = cv2.resize(cv2.imread(dirs_im[j]),(224, 224))
        x_train[p,:,:,:] = x/256. 
        y_train[p,i] = 1
        p += 1
        print(p)
        
        
# define cnn model
def define_model(classes):
    # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(classes, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = Adam(lr=1e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


model = define_model(classes)

    
# randomize data
index = np.arange(len(y_train))
np.random.shuffle(index)
x_train = x_train[index,:,:,:]
y_train = y_train[index]  

    
results = model.fit(x_train, y_train, validation_split=0.1, batch_size=32 ,epochs=5)
   
    






















model.save("C:\\Users\\moi\\Desktop\\formation_python_celec\\keras_models\\model03.h5")

