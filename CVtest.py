import getFeature
import loadData
import SGD
import SGDwithDT
from sklearn import tree

step_SGD = 0.02
err_th_SGD = 10e-8

step_SGDwithDT = 0.005
err_th_SGDwithDT = 10e-9

# get traindata file for CV
# 'training.data.for.fold1/2/3/4/5'
# correspond to test data ''badges.modified.data.fold1/2/3/4/5''
#getFeature.createDataSetForCV()
CV = 5
data_fileName_train = []
ARFF_fileNAme_train = []
data_fileName_test = []
ARFF_fileNAme_test = []
X_train,Y_train,fn_train = [],[],[]
X_test,Y_test,fn_test = [],[],[]
#for i in range(CV):
#    data_fileName_train.append('training.data.for.fold'+str(i+1))
#    data_fileName_test.append('badges.modified.data.fold'+str(i+1))
#    ARFF_fileNAme_train.append('training.data.fold%c.arff' %str(i+1))
#    ARFF_fileNAme_test.append('testing.data.fold%c.arff' %str(i+1))
#    getFeature.featureExtract(data_fileName_train[i],\
#                              ARFF_fileNAme_train[i],'trainData'+str(i+1))
#    getFeature.featureExtract(data_fileName_test[i],\
#                              ARFF_fileNAme_test[i],'testData'+str(i+1))
#    X_tmp,Y_tmp,fn_tmp = loadData.loadARFF2py(ARFF_fileNAme_train[i])
#    X_train.append(X_tmp)
#    Y_train.append(Y_tmp)
#    fn_train.append(fn_tmp)
#    X_tmp,Y_tmp,fn_tmp = loadData.loadARFF2py(ARFF_fileNAme_test[i])
#    X_test.append(X_tmp)
#    Y_test.append(Y_tmp)
#    fn_test.append(fn_tmp)

def crossValidation5():
    # SGD
    accSGD = []
    for i in range (CV):
        w,theta = SGD.SGD_clf(X_train[i],Y_train[i],step_size = step_SGD,threshold = err_th_SGD)
        accSGD.append(SGD.accScore_SGD(X_test[i],Y_test[i],w,theta))
    
    print 'SGD accuracy :'
    print accSGD
    pA_sgd = reduce(lambda x, y: x + y, accSGD) / float(len(accSGD))
    print 'pA_SGD = %f' %pA_sgd
    print '#----------------------------'
    
    #----------------------------
    # Full - DT
    accFDT = []
    for i in range(CV):
        clf = tree.DecisionTreeClassifier(criterion='entropy',random_state = 0)
        clf = clf.fit(X_train[i], Y_train[i])
        accFDT.append(clf.score(X_test[i],Y_test[i]))
    
    print 'Full DT accuracy :'
    print accFDT
    pA_fdt = reduce(lambda x, y: x + y, accFDT) / float(len(accFDT))
    print 'pA_FDT = %f' %pA_fdt
    print '#----------------------------'
    
    #----------------------------
    # len 4 DT
    accDT4 = []
    for i in range(CV):
        clf4 = tree.DecisionTreeClassifier(criterion='entropy',random_state = 0,max_depth = 4)
        clf4 = clf4.fit(X_train[i], Y_train[i])
        accDT4.append(clf4.score(X_test[i],Y_test[i]))
    
    print 'depth 4 DT accuracy :'
    print accDT4
    pA_dt4 = reduce(lambda x, y: x + y, accDT4) / float(len(accDT4))
    print 'pA_DT4 = %f' %pA_dt4
    print '#----------------------------'
    
    ##----------------------------
    ## len 8 DT
    #accDT8 = []
    #for i in range(CV):
    #    clf8 = tree.DecisionTreeClassifier(criterion='entropy',random_state = 0,max_depth = 8)
    #    clf8 = clf8.fit(X_train[i], Y_train[i])
    #    accDT8.append(clf8.score(X_test[i],Y_test[i]))
    #
    #print 'depth 8 DT accuracy :'
    #print accDT8
    #pA_dt8 = reduce(lambda x, y: x + y, accDT8) / float(len(accDT8))
    #print 'pA_DT8 = %f' %pA_dt8
    #print '#----------------------------'
    ##----------------------------
    #
    ## SGD with DTs
    #accSGD_DTs = []
    #for i in range(CV):
    #    score = SGDwithDT.SGD_DT(X_train[i],Y_train[i],X_test[i],Y_test[i],\
    #            stepSize=step_SGDwithDT,errorThreshold=err_th_SGDwithDT)
    #    accSGD_DTs.append(score)
    #
    #print 'SGD over DTs accuracy'
    #print accSGD_DTs
    #pA_sgd_dt = reduce(lambda x, y: x + y, accSGD_DTs) / float(len(accSGD_DTs))
    #print 'pA_DT8 = %f' %pA_sgd_dt
    #print '#----------------------------'
    ##----------------------------
    #return accSGD,accFDT,accDT4,accDT8,accSGD_DTs
    
    ####################
    ####    ###     ####
    ##       #        ##
    #                  #
    #                  #
    ##                ##
    ####            ####
    ######        ######
    ########    ########
    #########  #########
    ####################