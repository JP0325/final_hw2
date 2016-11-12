def label2num(c):
    if c == '+':
        return 1.0
    elif c == '-':
        return -1.0

def loadARFF2py(arffName):
    import numpy as np
    import arff
    import codecs
    
    fo = codecs.open(arffName, 'rb')
    d = arff.load(fo)
    
    #get the name of each features
    dt = d['attributes'][0:-1]
    fn = [str(i[0]) for i in dt] #just choose the first string, by i[0]
    
    # turn the data into array
    dd = np.array(d['data'])    
    
    #convert the unicode to float, which can be handled by sklearn
    #-----------X and Y will be the training data and labels from now!!!
    X = dd[:,0:-1].astype(np.float) 
    Y = np.array(map(label2num,dd[:,-1]))

    fo.close()
    return X,Y,fn
