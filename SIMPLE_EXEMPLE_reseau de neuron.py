 

import numpy as np




x = np.array([[4,2],
              [4,3],
              [5,3],
              [5,2],
              [2.93,5.21],
              [1.69,4.15],
              [1.47,2.44],
              [2.91,0.3],
              [5.16,-0.24],
              [6.69,0.21],
              [7.72,2.13],
              [7.09,4.08],
              [5.25,5.37]])

y = np.array([[0],
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
              [1]])

def nonlin_numpy(x,deriv=False):
    
    if deriv == True:
        return ( 1/(1+np.exp(-x)) )*(1 - 1/(1+np.exp(-x)) )
    else:
	    return 1/(1+np.exp(-x))
    

w1 = 2 * np.random.random((2,4)) - 1

w2 = 2 * np.random.random((4,1)) - 1

b1 = 2 * np.random.random((1,4)) - 1

b2 = 2 * np.random.random((1,1)) - 1


for i in range(10000):

    s1 = x.dot(w1) + b1
    
    out1 = nonlin_numpy(s1)
    
    s2 = out1.dot(w2) + b2
    
    out2 = nonlin_numpy(s2)
    
    
    
    e = 0.5*(out2-y)**2
    
    err = e.sum() / len(e)
    
    print(err)
    
    
    deriv1 = out2 - y
    deriv2 = nonlin_numpy(s2,deriv=True)
    deriv3 = out1
    
    delta1 = deriv3.T.dot( deriv1 * deriv2 )
    
    lr = 0.1
    
    w2 = w2 - delta1 * 0.1
    b2 = b2 - (deriv1 * deriv2).sum(axis=0) * 0.1
    
    
    
    
    deriv1 = (deriv1 * deriv2 ).dot(w2.T)
    deriv2 = nonlin_numpy(s1,deriv=True)
    deriv3 = x
    
    delta2 = deriv3.T.dot( deriv1 * deriv2 )
    
    w1 = w1 -  delta2 * 0.1
    b1 = b1 -  (deriv2 * deriv2).sum(axis=0) * 0.1






x = np.array([[4.71,2.55]])

s1 = x.dot(w1) + b1

out1 = nonlin_numpy(s1)

s2 = out1.dot(w2) + b2

out2 = nonlin_numpy(s2)


print("\n\n\n")
print("classe1: ",out2)




x = np.array([[6.17,4.44]])

s1 = x.dot(w1) + b1

out1 = nonlin_numpy(s1)

s2 = out1.dot(w2) + b2

out2 = nonlin_numpy(s2)


print("\n\n\n")
print("classe1: ",out2)






























