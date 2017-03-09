# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:08:56 2016

@author: Garima
"""
import numpy as np
import sys




def softmax(X,w):
#    num= np.exp(np.dot(X,w.T))
#    denom= np.sum(num, axis=1)
#    denom= denom.reshape(denom.shape[0],1)
#    val= num/denom
#    return val 

     e_x = np.exp(np.dot(X,w.T))+0.000001
     denom=np.sum(np.exp(np.dot(X,w.T)),axis=1)+0.000001
     val= np.divide(e_x,denom[...,np.newaxis])
     return val
    
def trainData(data,trainPercent,k,a,iterations):       
        N= data.shape[0]
        D= data.shape[1]-1 
        
        idxs=np.arange(N)
        np.random.shuffle(idxs)
        tr=80*N/100        
        trainIdx= idxs[0:tr]
        testIdx= idxs[tr:]
        training, test = data[trainIdx,:], data[testIdx,:]
        testError=[]
        for tp in trainPercent:
            w= np.zeros((k,D))
            totaln= training.shape[0]
            size= totaln*tp/100
            X= training[:size,0:-1]
            y= training[:size,-1]
            Y= np.zeros((size,k))
            for i,val in enumerate(y):
                Y[i,val]=1
            itr=0
            while(itr<iterations):
                p=softmax(X,w)
                p=p-Y
                grad= np.dot(p.T,X)
                grad=a*grad
                w= w - grad
                itr +=1
            
            #find the test error
            incorrect=0
            for t in range(0,test.shape[0]):
                label= np.argmax(np.dot(test[t,0:-1],w.T))
                if label != test[t,-1]:
                    incorrect= incorrect+1
            errorrate= incorrect/float(test.shape[0])
            testError.append(errorrate)
        return testError


def logisticRegression(filename,splits, trainPercent):
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
        a=0.0000001
        iterations=10000
        for s in range(0,splits):
            #train data        
            error50=trainData(boston50,trainPercent,2,a,iterations )
            E50.append(error50)
            print('Boston50')
            print('For split ', s )
            for e in error50:
                print (e*100 )
            print("are error rates for 10,25,50,75 and 100 percent respectively")
        for s in range(0,splits):
            #train data
            error75=trainData(boston75,trainPercent,2,a,iterations )
            E75.append(error75)
            print('Boston75')
            print('For split ', s )
            for e in error75:
                print (e*100 )
            print("are error rates for 10,25,50,75 and 100 percent respectively")
            
         #plot graph of  testerror vs trainpercent
        sd50=np.std(E50, axis=0)
        sd75=np.std(E75, axis=0)
        E50=np.sum(E50, axis=0)/float(splits)
        E75=np.sum(E75, axis=0)/float(splits)
        print('boston50 Mean Error %:',E50*100)
        print('boston75 Mean Error %: ',E75*100)
        print('boston50 standard deviation %:',sd50)
        print('boston75 standard deviation %:',sd75)
#        plt.plot(trainPercent,E50, label='boston50') 
#        plt.plot(trainPercent,E75, label='boston75') 
#        plt.show()
        
    elif 'digits' in filename:
        E=[]    
        a=0.00001
        iterations=1000
        for s in range(0,splits):
            error= trainData(data,trainPercent,10,a,iterations)
            E.append(error)
            print('Digit')
            print('For split ', s )
            for e in error:
                print (e*100 )
            print("are error rates for 10,25,50,75 and 100 percent respectively")
        sd=np.std(E, axis=0)
        E=np.sum(E, axis=0)/float(splits)
        print("Mean error percent:",E*100)
        print('Digits standard deviation %:',sd)
#        plt.plot(trainPercent,E, label='digits') 
#        plt.show()
        
        
filename= sys.argv[1]
splits= int(sys.argv[2])
trainPercent=[]
for i in range(3,len(sys.argv)):
    trainPercent.append(int(sys.argv[i]))
logisticRegression(filename,splits,trainPercent)