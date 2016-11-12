import CVtest
import numpy as np

t = 4.604
accSGD,accFDT,accDT4,accDT8,accSGD_DTs = CVtest.crossValidation5()

pA_DT4 = sum(accDT4) / len(accDT4)
pA_DT8 = sum(accDT4) / len(accDT4)
pA_FDT = sum(accFDT) / len(accFDT)
pA_SGD = sum(accSGD) / len(accSGD)
pA_SGD_DTs = sum(accSGD_DTs) / len(accSGD_DTs)

D_DT4 = sum((pA_DT4 - value) ** 2 for value in accDT4) / (len(accDT4)-1)
D_DT8 = sum((pA_DT8 - value) ** 2 for value in accDT8) / (len(accDT8)-1)
D_FDT = sum((pA_FDT - value) ** 2 for value in accFDT) / (len(accFDT)-1)
D_SGD = sum((pA_SGD - value) ** 2 for value in accSGD) / (len(accSGD)-1)
D_SGD_DTs = sum((pA_SGD_DTs - value) ** 2 for value in accSGD_DTs) / (len(accSGD_DTs)-1)

Sn_DT4 = D_DT4**0.5
Sn_DT8 = D_DT8**0.5
Sn_FDT = D_FDT**0.5
Sn_SGD = D_SGD**0.5
Sn_SGD_DTs = D_SGD_DTs**0.5
sqrtn = len(accDT4)**0.5


print '99% confidence interval of pA_DT4:',\
        [pA_DT4-t*Sn_DT4/sqrtn, pA_DT4+t*Sn_DT4/sqrtn]
print 'std is ',Sn_DT4

print '99% confidence interval of pA_DT8:',\
        [pA_DT8-t*Sn_DT8/sqrtn, pA_DT8+t*Sn_DT8/sqrtn]
print 'std is ',Sn_DT8

print '99% confidence interval of pA_FDT:',\
        [pA_FDT-t*Sn_FDT/sqrtn, pA_FDT+t*Sn_FDT/sqrtn]
print 'std is ',Sn_FDT
        
print '99% confidence interval of pA_SGD:',\
        [pA_SGD-t*Sn_SGD/sqrtn, pA_DT8+t*Sn_SGD/sqrtn]
print 'std is ', Sn_SGD

print '99% confidence interval of pA_SGD_DTs:',\
        [pA_SGD_DTs-t*Sn_SGD_DTs/sqrtn, pA_SGD_DTs+t*Sn_SGD_DTs/sqrtn]
print 'std is ', Sn_SGD_DTs



def differenceTest(a1,a2):
    p1 = np.array(a1)
    p2 = np.array(a2)
    deltaP = p1 - p2
    ts = deltaP.mean()/ (deltaP.std(ddof=1) * (len(deltaP)**2))
    print ts
    t90 = 2.132
    t95 = 2.776
    t99 = 4.604
    if abs(ts) >= t99:
        print '99% different.'
    elif abs(ts) >= t95:
        print '95% different'
    elif abs(ts) >= t90:
        print '90% different'
    else:
        print 'no significant difference'

differenceTest(accSGD_DTs,accFDT)
differenceTest(accSGD_DTs,accSGD)
differenceTest(accFDT,accDT8)
differenceTest(accFDT,accDT4)
differenceTest(accDT8,accDT4)
differenceTest(accDT4,accSGD)
    
    