import numpy as np
import time



def nonlin_numpy(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

def train_dnn_numpy(x_in,y_in,nb_hiden,epoc,lr,batch_size):
    
    op_n = len(nb_hiden)-2
    
    # randomly initialize our weights with mean 0
    w = []
    for i in range(op_n+1):
        w.append(2*np.random.random((nb_hiden[i],nb_hiden[i+1])) - 1)
    b = []
    for i in range(op_n+1):
        b.append(2*np.random.random((1,nb_hiden[i+1])) - 1)
    
    
    for j in range(epoc):
        
        e_g = 0
    
        for k in range(len(x_in)//batch_size):
            
            x = x_in[k*batch_size:k*batch_size+batch_size,:]
            y = y_in[k*batch_size:k*batch_size+batch_size,:]
            
            # Feed forward through layers
            c_layers = []
            c_layers.append(x)
            
            for i in range(op_n+1):
                c_layers.append(nonlin_numpy(np.dot(c_layers[i],w[i])+b[i]))
                
            e = y - c_layers[op_n+1]
            e_g += e
            
            error = []
            delta = []
            
            
            error.append(e)
            delta.append(e * nonlin_numpy(c_layers[op_n+1],deriv=True))
            
            for i in range(op_n):
                error.append((delta[i]).dot((w[op_n-i]).T))
                delta.append(error[i+1] * nonlin_numpy(c_layers[op_n-i],deriv=True)) 
            
            for i in range(op_n+1):
                w[i] += (c_layers[i]).T.dot(delta[op_n-i])*lr
                b[i] += (delta[op_n-i]).sum(axis=0)*lr
                
                
        if (j% 10) == 0:
            print(" ")
            print( "Error:" + str(e_g.sum()))
        
    return w,b




def predict_dnn_numpy(x,nb_hiden,w,b):
    k = x
    for i in range(len(nb_hiden)-1):
        k = nonlin_numpy(np.dot(k,w[i])+b[i])
    return k




from keras.datasets import mnist
from keras.utils import to_categorical

# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()

# convert target data to categories
target_train = to_categorical(target_train)
target_test  = to_categorical(target_test)


# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test  = input_test / 255
    
# reshape data
input_train = input_train.reshape(len(input_train),784)
input_test  = input_test.reshape(len(input_test),784)


# a = (input_train[100,:]).reshape(28,28)
# %varexp --imshow a

# size of network
nb_hiden = [784,60,60,60,10]

epoc=500
lr = 0.001
batch_size = 16

w,b = train_dnn_numpy(input_train,target_train,nb_hiden,epoc,lr,batch_size)



# from save_w_NB1 import select_data_from_data_base
# w = select_data_from_data_base("w.db")
# b = select_data_from_data_base("b.db")


r = predict_dnn_numpy(input_train,nb_hiden,w,b)
true = 0
for i in range(60000):
    if np.argmax(r[i,:]) == np.argmax(target_train[i,:]):
        true += 1       
print("accuracy(target_train): ",round((true/60000)*100,2),"%")



r = predict_dnn_numpy(input_test,nb_hiden,w,b)
true = 0
for i in range(10000):
    if np.argmax(r[i,:]) == np.argmax(target_test[i,:]):
        true += 1      
print("\naccuracy(target_test): ",round((true/10000)*100,2),"%")
















