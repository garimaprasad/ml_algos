# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 22:42:48 2016

@author: Garima
"""
import numpy as np
import time
import sys
from random import randint


filename= sys.argv[1]
data = np.genfromtxt(filename, delimiter=',')
d = data.shape[1] - 1
n = data.shape[0]
data[data[:,0]==1,0]=-1  
data[data[:,0]==3,0]=1

runs= int(sys.argv[3])
lamb=1
MValue= sys.argv[2].split(",")
M= map(int,MValue)

##########################################
def calculateObjective(lamb,w,Y,X):
    innerProd= np.dot(X,w.T)
    yPred=np.multiply(innerProd,Y[:, np.newaxis])
    objective=lamb*0.5*np.power(np.linalg.norm(w),2) + (1/float(n))*np.sum(max(0, (1-yp)) for yp in yPred)
    return objective

##########################################
def calculateATPlus(At,w):
    ii=[]
    for i in range(0,len(At)):
        if np.multiply( At[i,0], (np.dot(At[i,1:],w.T)) ) >= 1:
            ii.append(i)    
    AtPlus= np.delete(At, (ii), axis=0)
    return AtPlus

##########################################
def calGradient(data,w):
    AtPlus= calculateATPlus(data,w)
    x=AtPlus[:,1:]
    y=AtPlus[:,0]
    y=y.reshape(len(y),1)
    yx=np.sum(np.multiply(y,x), axis=0)
    grad= w - np.divide(yx , float(data.shape[0]))
    return grad, AtPlus.shape[0]

##########################################    
def getRandomIndx(x):
    return randint(0, x.shape[0] - 1)

##########################################    
def epoch(w,m,mu, subgrade):
    lRate= 0.0001
    wbar=w
    for t in range(0,m):
        i=getRandomIndx(data)
        gradW,n2 = calGradient(data[i].reshape(1,data[i].shape[0]),w)
        gradWbar,n3 = calGradient(data[i].reshape(1,data[i].shape[0]),wbar)
        subgrade += n2+n3
        w= w - np.multiply(lRate,(gradW -gradWbar +mu) )   
    return w, subgrade
    
##########################################  
for m in M :
    times=[]
    for run in range(runs):
        xis=data[:, 1:]
        yis=data[:,0]        
        w= np.ones((1,d))
        computations= 100*n
        objs=[]
        s=1
        subGrade=0
        start=time.time()
        while(subGrade<= computations and s<=1000):            
            mu,n1= calGradient(data,w)
            subGrade += n1
            w, subGrade = epoch(w,m,mu, subGrade)
            s +=1
        end=time.time()
        times.append(end-start)
    print'Epoch size ', m
    meanTime = np.mean(times, axis=0)
    sd=np.std(times, axis=0)
    print 'average time' , meanTime
    print 'standard deviation', sd
    