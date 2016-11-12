import SGD
import loadData
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import SGDClassifier
from sklearn import tree

def tuneSGDofDT(stepSize,errorThreshold):
    '''
    tuneSGDofDT(0.005,10e-9) is even better
    acc1 =  0.728813559322  step =  0.005
    optimal =  0.677966101695
    whole =  0.867346938776

    '''
    X,Y,fn = loadData.loadARFF2py('badges_all.arff')
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split\
                                        (X, Y, test_size=0.2, random_state=0)
    
    numTree = 100
    SR = 0.5   #sample rate
    
    numExample = len(X_train)
    numTest = len(X_test)
    # create a 2D array newFeature to store the output of each DT
    # newFeature is 294*100 (at most, all 294 data and 100 tree)
    newFeature = np.zeros(shape=(numExample,numTree))
    X_test_transform = np.zeros(shape=(numTest,numTree))
    
    for i in range(numTree):
        # for each tree, i gonna sample by different seed(i),
        # and set the seed can make this script repeatable by anytime
        # each time just pick 147 different examples to train the DT
        np.random.seed(i)    
        randPick = np.random.choice(numExample,size = numExample*SR,replace = False)
        X_trainForTree = X_train[randPick]
        y_trainForTree = y_train[randPick]
        # train small tree with sampled data
        clf = tree.DecisionTreeClassifier(criterion='entropy',random_state = i,max_depth = 4)   
        clf = clf.fit(X_trainForTree, y_trainForTree)
        for n in range(numExample):
            lab_output_train = int(clf.predict([X_train[n]]))
            if lab_output_train == 1: 
                newFeature[n,i] = lab_output_train
        # need to transfer the feature of test data, just let them go through DTs        
        for m in range(numTest):
            lab_output_test = int(clf.predict([X_test[m]]))
            if lab_output_test == 1:
                X_test_transform[m,i] = lab_output_test
    
    
    w, theta = SGD.SGD_clf(newFeature,y_train,step_size = stepSize,threshold = errorThreshold)
    acc = SGD.accScore_SGD(X_test_transform,y_test,w,theta)
    print 'acc1 = ',acc, ' step = ',stepSize
    
    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(newFeature, y_train)
    acc2 = clf.score(X_test_transform,y_test)
    acc3 = clf.score(np.vstack([newFeature,X_test_transform]),np.append(y_train,y_test))
    
    print 'optimal = ',acc2
    print 'whole = ',acc3
    print '---------------------'

#for n in range(5):
#    tuneSGDofDT(stepSize[0][n],10e-5)
#    
#acc1 =  0.610169491525  step =  0.01
#optimal =  0.847457627119
#whole =  0.87074829932
#---------------------
#acc1 =  0.508474576271  step =  0.02
#optimal =  0.779661016949
#whole =  0.826530612245
#---------------------
#run out of iteration
#acc1 =  0.406779661017  step =  0.03
#optimal =  0.694915254237
#whole =  0.775510204082
#---------------------
#run out of iteration
#acc1 =  0.406779661017  step =  0.04
#optimal =  0.694915254237
#whole =  0.775510204082
#---------------------
#run out of iteration
#acc1 =  0.406779661017  step =  0.05
#optimal =  0.694915254237
#whole =  0.775510204082
#---------------------
#========================================
#th = 10.**(-1*np.array(range(5,10)))
#for n in range(5):
#    tuneSGDofDT(0.01,th[n])
#    
#acc1 =  0.610169491525  step =  0.01
#optimal =  0.847457627119
#whole =  0.87074829932
#---------------------
#acc1 =  0.610169491525  step =  0.01
#optimal =  0.847457627119
#whole =  0.87074829932
#---------------------
#acc1 =  0.779661016949  step =  0.01
#optimal =  0.847457627119
#whole =  0.877551020408
#---------------------
#acc1 =  0.779661016949  step =  0.01
#optimal =  0.847457627119
#whole =  0.877551020408
#---------------------
#acc1 =  0.847457627119  step =  0.01
#optimal =  0.796610169492
#whole =  0.843537414966
#---------------------


