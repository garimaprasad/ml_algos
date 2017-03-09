# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 20:56:05 2016

@author: Garima
"""
import numpy as np
#import matplotlib.pyplot as plt
import sys

def gaussian(val, m, var):
    m=m+0.000001
    var=var+0.000001
    num= np.exp((-0.5)*(np.power((val-m),2))/var)
    denom= np.sqrt(2*np.pi*var)
    pdf=num/denom
    return pdf
    
def trainData(data,trainPercent,k):       
        N= data.shape[0]
        D= data.shape[1]-1 
        #w= np.zeros((k,D))
        idxs=np.arange(N)
        np.random.shuffle(idxs)
        tr=80*N/100        
        trainIdx= idxs[0:tr]
        testIdx= idxs[tr:]
        training, test = data[trainIdx,:], data[testIdx,:]
        testError=[]
        for tp in trainPercent:
            totaln= training.shape[0]
            size= totaln*tp/100
            X= training[:size,0:-1]
            y= training[:size,-1]
            Y= np.zeros((size,k))
            for i,val in enumerate(y):
                Y[i,int(val)]=1
            #naive bayes
            prior=[]
            classCount=np.sum(Y, axis=0)
            cMu=[]
            cVar=[]
            for c in range(0,k):
                #print (classCount)
                p = classCount[c] / float(X.shape[0])
                prior.append(np.log(p))
                d= training[:size,:]
                d=d[d[:,-1]==c]
                mui=np.sum(d, axis=0)/float(classCount[c])
                mui=mui[0:-1]
                var=[]
                for i in range(0,D):
                    s=0;
                    for n,dn in enumerate(d[:,i]):
                       s=s+((dn-mui[i])**2)
                    var.append(s/float(classCount[c]-1))
                cMu.append(mui)
                cVar.append(var)                                                
                
            #find the test error
            incorrect=0
            for t in range(0,test.shape[0]):
                ak=[]
                for c in range(0,k):
                    sumPi=0
                    for i in range(0,D): 
                       pi=gaussian(test[t,i],cMu[c][i],cVar[c][i])
                       if np.isnan(pi):
                           pi = 0.000001
                           pi=np.log(pi)
                       else:
                          pi=np.log(pi) 
                       sumPi=sumPi+pi
                    ak.append(sumPi+prior[c]) 
                label= np.argmax(ak)
                if label != test[t,-1]:
                    incorrect= incorrect+1
            errorrate= incorrect/float(test.shape[0])
            testError.append(errorrate)
        return testError



def naiveBayesGaussian(filename,splits, trainPercent):
    # reading and preprocessing data
    data= np.genfromtxt(filename , delimiter=',')
    if 'boston' in filename:    
        boston50 = np.genfromtxt(filename , delimiter=',')
        boston75 = np.genfromtxt(filename , delimiter=',')
        median= np.percentile(boston50[:,13], 50)
        thirdquart= np.percentile(boston75[:,13], 75)
        boston50[boston50[:, 13] < median, 13] = 0
        boston50[boston50[:, 13] >= median, 13] = 1
        boston75[boston75[:, 13] < thirdquart, 13] = 0
        boston75[boston75[:, 13] >= thirdquart, 13] = 1
        E50=[]
        E75=[]
        for s in range(0,splits):
            #train data
            error50=trainData(boston50,trainPercent,2 )
            E50.append(error50)
            print('Boston50')
            print('For split ', s )
            for e in error50:
                print (e*100 )
            print("are error rates for 10,25,50,75 and 100 percent respectively")
        for s in range(0,splits):
            #train data
            error75=trainData(boston75,trainPercent,2 )
            E75.append(error75)
            print('Boston75')
            print('For split ', s )
            for e in error75:
                print (e*100 )
            print("are error rates for 10,25,50,75 and 100 percent respectively")
        
         #plot graph of  testerror vs trainpercent
        sd50=np.std(E50, axis=0)
        sd75=np.std(E75, axis=0)
        m50=np.sum(E50, axis=0)/float(splits)
        m75=np.sum(E75, axis=0)/float(splits)
        
        print('boston50 Mean Error %:',m50*100)
        print('boston75 Mean Error %: ',m75*100)
        print('boston50 standard deviation %:',sd50)
        print('boston75 standard deviation %:',sd75)
        

    elif 'digits' in filename:
        E=[]
        for s in range(0,splits):
            error= trainData(data,trainPercent,10)
            E.append(error)
            print('Digit')
            print('For split ', s )
            for e in error:
                print (e*100 )
            print("are error rates for 10,25,50,75 and 100 percent respectively")
        sd=np.std(E, axis=0)    
        m=np.sum(E, axis=0)/float(splits)
        print("Digits Mean error percent:",m*100)
        print('Digits standard deviation %:',sd)

        
filename= sys.argv[1]
splits= int(sys.argv[2])
trainPercent=[]
for i in range(3,len(sys.argv)):
    trainPercent.append(int(sys.argv[i]))
naiveBayesGaussian(filename,splits,trainPercent)