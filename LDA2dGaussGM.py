# -*- coding: utf-8 -*-

import numpy as np
import sys

filename = sys.argv[1]
num_crossval= int(sys.argv[2])
data= np.genfromtxt(filename, delimiter=',')

trainerror=[]
testerror=[]
foldsize= len(data)/num_crossval

def getKey(item):
    return item[0]

def gaussPdf(val, mn, cov):
    temp=np.dot((val-mn),np.linalg.inv(cov))
    num = np.exp(np.dot(temp,(val-mn).T)*(-0.5))
    #n=2
    denom= (2*np.pi)*np.sqrt(np.linalg.det(cov))
    #print(num*denom)
    pdf=num/denom
    return pdf
    
def finderror(projD,oData):
    incorrect=0
    for idx,x in enumerate(projD):
        likelihoods=[]
        for c in range(0,10):
            likelihoods.append(prior[c] * gaussPdf(x, xbar[c], sd[c]))
        label = np.argmax(likelihoods)
        if label != oData[idx,64]:
            incorrect= incorrect+1
    errorrate= incorrect/float(projD.shape[0])
    return errorrate
    
    
for k in range(num_crossval):
    
    testing = data[k*foldsize:(k+1)*foldsize]
    a=data[:k*foldsize]
    b= data[(k+1)*foldsize:]
    training = np.concatenate(( a , b))
    
    mean = []
    sw=0
    M=np.mean(training[:,0:-1], axis=0).reshape(64,1)
    for l in range(0,10):
        x= training[training[:, 64]==l]
        x=x[:,0:-1]
        m=(np.mean(x, axis=0)).reshape(64,1)
        m=m.T
        tempsw=0;
        for i in range(0,len(x)):
            diff= x[i,:]-m
            tempsw= tempsw + diff*diff.T
        mean.append(m)
        sw=sw+tempsw
        
    sb=0
    for i,m in enumerate(mean):  
        n = training[training[:, 64]==i].shape[0] 
        sb =sb + n * np.dot((m.T - M),(m.T - M).T)
    
    
    swInv= np.linalg.pinv(sw)
    eigVals, eigVecs = np.linalg.eig(np.dot(swInv,sb))
    eigPairs = [(np.abs(eigVals[i]), eigVecs[:,i]) for i in range(0,len(eigVals))]
    eigPairs = sorted(eigPairs, key=getKey, reverse=True) 
    
    w = np.hstack(( eigPairs[0][1].reshape(64,1) , eigPairs[1][1].reshape(64,1)))
    
    
    #mean of the dimensionally reduced data
    xbar=[]
    sd=[]
    prior=[]
    
    #gaussian distribution of claases
    for l in range(0,10):
        x= training[training[:, 64]==l]
        x=x[:,0:-1]
        projX= np.dot(x,w)
        p =projX.shape[0] / float(training.shape[0])
        prior.append(p)
        m=(np.mean(projX, axis=0)).reshape(2,1)
        m=m.T
        xbar.append(m)
        covar=np.cov(projX, rowvar=False) 
        sd.append(covar)
        
    projTrainData=  np.dot(training[:,0:-1],w)
    projTestData=  np.dot(testing[:,0:-1],w)
    
    etrain= finderror(projTrainData,training)
    trainerror.append(etrain)
        
    etest= finderror(projTestData,testing)
    testerror.append(etest)       
   
avgtesterror= np.sum(testerror)/float(num_crossval)
testSd= np.sqrt(np.sum((x- avgtesterror)**2 for x in testerror)/float(10))
avgtrainerror= np.sum(trainerror)/float(num_crossval)
trainSd=np.sqrt(np.sum((x- avgtrainerror)**2 for x in trainerror)/float(10))


print('test error percent: ',avgtesterror*100, 'train error percent: ',avgtrainerror*100)
print('test standard deviation: ',testSd, 'train standard deviation  ',trainSd)
