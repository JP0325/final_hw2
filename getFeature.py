'''
this Script is use to extract features from raw data and create a ARFF file
run featureExtraction.py <raw_data_file> <output_arff_file> <relationName>
'''
alp = 'abcdefghijklmnopqrstuvwxyz'

def Alpabet2Array(c):
    '''
    take a single string c as input
    return a array with a length of 26
    if c is '',which means the length of name is less than 5 word
    return zeros[1,26]
    '''
    z0 = [0]*26
    if c == '*':
        return z0
    else:
        for i in range(26):
            if c == alp[i]:
                z0[i] = 1
                return z0
    return ['*']*26

#----------
def featureExtract(raw_data_file,ouput_arff_file,relationName):
    import sys   
    #open a output file fout, which is an ARFF file
    fout = open(ouput_arff_file,'wb')
    
    ##----the header session
    fout.write('@relation '+relationName)
    fout.write('\n\n')
    #----the attribute name and type session---
    for Norder in ['first','last']:
        for lo in map(str,range(5)):
            for tmp in alp:
                attriString = '@attribute '+Norder +'Name' + lo + '='\
                            + tmp + ' {1,0}\n'
                fout.write(attriString)
    fout.write('@attribute Class {+,-}\n\n')
    
    #----now transform the data
    fout.write('@data\n')
    
    f = open(raw_data_file,'rb')
    for line in f:
        dataName = []
        if line == "":
            break
        else:
            labelofName = line.split()[0]
            firstName = line.split()[1]
            lastName = line.split()[2]
            for typeName in [firstName,lastName]:
                while len(typeName)<5:
                    typeName += '*'
                typeName = typeName[0:5]   
                for c in typeName:
                    dataName += Alpabet2Array(c)
            NameFeature = map(str,dataName)
            for ch in NameFeature:
                fout.write(ch + ',')
            fout.write(labelofName+'\n')
    
    fout.close()
    f.close()
    
def createDataSetForCV():
    c,co,f,d = [],[],[],[]
    for i in range(5):
        c.append('badges.modified.data.fold'+str(i+1))
        co.append('training.data.for.fold'+str(i+1))
        f.append(open(c[i]))
        d.append(f[i].read())
    fout1 = open(co[0],'w')
    fout2 = open(co[1],'w') 
    fout3 = open(co[2],'w') 
    fout4 = open(co[3],'w') 
    fout5 = open(co[4],'w')
    fout1.write(d[1]+d[2]+d[3]+d[4])
    fout2.write(d[0]+d[2]+d[3]+d[4])
    fout3.write(d[0]+d[1]+d[3]+d[4])
    fout4.write(d[0]+d[1]+d[2]+d[4])
    fout5.write(d[0]+d[1]+d[2]+d[3])
    for i in range(5):
        f[i].close()
    fout1.close()
    fout2.close()
    fout3.close()
    fout4.close()
    fout5.close()
    
