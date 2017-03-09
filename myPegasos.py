# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 22:42:48 2016

@author: Garima
"""
import numpy as np
import time
import sys

filename= sys.argv[1]
kValue= sys.argv[2].split(",")
K= map(int,kValue)
runs= int(sys.argv[3])

data = np.genfromtxt(filename, delimiter=',')
d = data.shape[1] - 1
n = data.shape[0]
x1=data[data[:,0]==1]
x2=data[data[:,0]==3]
lamb=1

#############################################
def calculateObjective(lamb,w,Y,X):
    innerProd= np.dot(X,w.T)
    yPred=np.multiply(innerProd,Y[:, np.newaxis])
    objective=lamb*0.5*np.power(np.linalg.norm(w),2) + (1/float(n))*np.sum(max(0, (1-yp)) for yp in yPred)
    return objective
    
#############################################    
def getClassData(x,kpercent):
    idxs=np.arange(x.shape[0])
    np.random.shuffle(idxs) 
    n1=int(x.shape[0]*kpercent)      
    x1dx= idxs[0:n1]    
    xClass= x[x1dx,:]
    return xClass

#############################################    
def doPegasos(w,k,t,subGrad):
    kpercent= k/float(n)   
    
    if k==1:
        At=getClassData(data,kpercent)
    elif k==n:
        At=data
    else:
        c1= getClassData(x1,kpercent)
        c2= getClassData(x2,kpercent)
        At=np.concatenate((c1,c2))  
    At[At[:,0]==3,0]=-1
    ii=[]
    for i in range(0,len(At)):
        if np.multiply( At[i,0], (np.dot(At[i,1:],w.T)) ) >= 1:
            ii.append(i)
    At= np.delete(At, (ii), axis=0)
    subGrad += At.shape[0]
    x=At[:,1:]
    y=At[:,0]
    y=y.reshape(len(y),1)
    wi=np.multiply((1-(1/float(t*lamb))),w) + np.multiply((1/float(k*t*lamb)), 
                    np.sum(np.multiply(y,x), axis=0))
    winorm=np.linalg.norm(wi)
    temp= 1/float(np.sqrt(lamb) * winorm)
    temp=min(1,temp)
    w= np.multiply(temp,wi)
    return w,subGrad
    
#############################################   
for k in K :
    times=[]
    for run in range(runs):
        xis=data[:, 1:]
        yis=data[:,0]
        yis[yis[:] == 3] = -1
        start=time.time()
        w= np.zeros((1,d))
        computations= 100*n
        objs=[]
        subGrad=0
        t=1
        while subGrad<= computations and t<=1000:
            w,subGrad=doPegasos(w,k,t,subGrad)
            t +=1
        end=time.time()        
        times.append(end-start)
    print'Minbatch size ', k
    meanTime = np.mean(times, axis=0)
    sd=np.std(times, axis=0)
    print 'average time' , meanTime
    print 'standard deviation', sd