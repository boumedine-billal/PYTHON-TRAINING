import numpy as np
import time

def nonlin_numpy(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

def train_dnn_numpy(x,y,nb_hiden,epoc,lr):
    
    op_n = len(nb_hiden)-2
    
    # randomly initialize our weights with mean 0
    w = []
    for i in range(op_n+1):
        w.append(2*np.random.random((nb_hiden[i],nb_hiden[i+1])) - 1)
    b = []
    for i in range(op_n+1):
        b.append(2*np.random.random((1,nb_hiden[i+1])) - 1)
    
    
    for j in range(epoc):
    
        # Feed forward through layers
        c_layers = []
        c_layers.append(x)
        
        for i in range(op_n+1):
            c_layers.append(nonlin_numpy(np.dot(c_layers[i],w[i])+b[i]))
            
        e = y - c_layers[op_n+1]
        error = []
        delta = []
        
     
        if (j% 100) == 0:
            print(" ")
            print( "Error:" + str(e.sum()))
        
        error.append(e)
        delta.append(e * nonlin_numpy(c_layers[op_n+1],deriv=True))
        
        for i in range(op_n):
            error.append((delta[i]).dot((w[op_n-i]).T))
            delta.append(error[i+1] * nonlin_numpy(c_layers[op_n-i],deriv=True)) 
        
        for i in range(op_n+1):
            w[i] += (c_layers[i]).T.dot(delta[op_n-i])*lr
            b[i] += (delta[op_n-i]).sum(axis=0)*lr
            
        
    return w,b

def predict_dnn_numpy(x,nb_hiden,w,b):
    k = x
    for i in range(len(nb_hiden)-1):
        k = nonlin_numpy(np.dot(k,w[i])+b[i])
    return k






x = np.array([[3,2],
              [20,1],
              [20,2],
              [19,1],
              [19,2],
              [5,2],
              [7,2],
              [2,3],
              [8,3],
              [2,5],
              [8,5],
              [2,7],
              [8,7],
              [3,8],
              [5,8],
              [7,8],
              [4,4],
              [5,4],
              [6,4],
              [4,5],
              [5,5],
              [6,5],
              [4,6], 
              [5,6],             
              [6,6]])/10.

y = np.array([[1],
              [0],
              [0],
              [0],
              [0],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0], 
              [0],
              [0]])
    


# size of network
nb_hiden = [2,20,20,4,1]


epoc=200000
lr = 0.1


w,b = train_dnn_numpy(x,y,nb_hiden,epoc,lr)



r = predict_dnn_numpy(x,nb_hiden,w,b)









