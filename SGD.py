# -*- coding: utf-8 -*-
'''
the randomState is set to seed == 0, so this can be repeatable.
'''
import numpy as np
from sklearn.metrics import accuracy_score

def SGD_clf(X,Y,Itertime = 100000,step_size = 0.01,threshold = 0.01):
    numFeature = len(X[0])
    numExample = len(X)
    w_0 = np.ones((numFeature,1))
    Y = [1 if y==1 else -1 for y in Y] #make sure that Y is label by {+1,-1}
    w = np.append(w_0, 1)
    np.random.seed(0)
    for i in range(Itertime):
        randPick = np.random.randint(numExample,size = 1,)
        x = np.append(X[randPick],1)
        y = Y[randPick]
        loss = 1./2. * (y - np.dot(x,w))**2
        if loss <= threshold:
            break
        delta_w =step_size*(y - np.dot(x,w))*x
        w += delta_w
    if i == Itertime - 1:
        print 'run out of iteration'
    return w[0:-1],w[-1]
    
#def accScore_SGD(X_test,Y_test,w,theta):
#    #make sure that Y is label by {+1,-1}
#    Y_test = [1 if y==1 else -1 for y in Y_test]
#    
#    h_output = X_test.dot(w)+theta
#    h_label = [1 if  x>= 0 else -1 for x in h_output]
#    return accuracy_score(h_label,Y_test)