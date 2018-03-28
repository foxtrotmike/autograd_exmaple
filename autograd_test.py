# -*- coding: utf-8 -*-
"""
Created on Thu Mar 08 22:22:16 2018
 ,
@author: afsar
"""

import autograd.numpy as np
from autograd import grad
    
def primal(weights):
    """
    This is the primal objective function
    """
    w = weights[:-1]
    b = weights[-1]
    score = np.dot(inputs,w)+b#    
    obj = 0.5*lambdaa*np.linalg.norm(w)**2
    loss = np.linalg.norm(score-targets)**2
    #np.mean(np.max(np.vstack((np.zeros(z.shape),1-z)),axis = 0))
    #z = targets*score
#    loss = np.mean((z<=0)*(0.5-z)+(z>0)*(z<1)*0.5*(1-z)**2)
    obj+=loss
    return obj#np.mean((np.dot(inputs,weights)-targets)**2)

if __name__=='__main__':
    # Setup some training data
    inputs = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
    targets = np.array([-1,1,1,1])
    
    lambdaa = 0.01 #regularization parameter
    lrate = 0.01 #learning rate
    T = 100 #number of epochs
    # Define a function that returns gradients of training loss using Autograd.
    training_gradient_fun = grad(primal)
    L = []
    # Optimize weights using gradient descent.
    weights = np.random.rand(inputs.shape[1]+1)
    print("Initial loss:", primal(weights))    
    for i in range(T):
        weights -= training_gradient_fun(weights) * lrate
        L.append( primal(weights))
    
    w = weights[:-1]
    b = weights[-1]
    score = np.dot(inputs,w)+b    
    print "Classification Accuracy",np.mean((2*(score>0)-1)==targets)
    import matplotlib.pyplot as plt
    plt.plot(L); 
    plt.xlabel('iterations'); plt.ylabel('Loss'); plt.grid()
    plt.show()