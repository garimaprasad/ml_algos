# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import sys 
import math 
from math import log


def create_tree(ind,training):
    feat_split= np.zeros((3,2))*(-1)
    feat_split[0,0], feat_split[0,1]= create_node(ind,training,[np.inf])
    indL,indR= get_node_points(feat_split[0,0], feat_split[0,1],training,ind)
    if(len(indL)>0): 
        feat_split[1,0],feat_split[1,1]= create_node(indL,training,[feat_split[0,0]])
    if(len(indR)>0):
        feat_split[2,0],feat_split[2,1]= create_node(indR,training,feat_split[0:2,0])
   
    bin1,bin2=get_node_points(feat_split[1,0],feat_split[1,1],training,indL)
    bin3,bin4=get_node_points(feat_split[2,0],feat_split[2,1],training,indR)    
    bin_label= np.zeros(4, dtype=np.int)*(-1)
    if len(bin1)>0:
        bin_label[0]= get_label(bin1,training )
    if len(bin2)>0:    
        bin_label[1]= get_label(bin2,training )
    if len(bin3)>0:
        bin_label[2]= get_label(bin3,training )
    if len(bin4)>0:
        bin_label[3]= get_label(bin4,training )
    return feat_split,bin_label
    
    
def get_node_points(feature, split_value,x2,indices):
    left=[]
    right=[]
    for j in indices:
        if(x2[j,int(feature)] <= split_value ):
            left.append(j)
        else:
            right.append(j)
    return left,right
   
def get_label(indices,training ):
    label=0
    x1=training[indices,13]
    c0 = x1[x1==0].shape[0]
    c1 = x1[x1==1].shape[0]
    if c0 <= c1:
        label=1
    return label
    
    
def create_node(index,training,exclude):
    
    per_split= [10,20,30,40,50,60,70,80,90]
    #for each attribute find best split
    split= np.zeros(training.shape[1]-1)
    info= np.zeros(training.shape[1]-1)
    for i in range(training.shape[1]-1):
        minInfo= np.inf
        minS=0
        if i not in exclude:           
            if i==3 :
                classify=np.chararray((training[index].shape[0],1))
                labels=training[index,13]               
                classify[training[index, i] == 0, 0] = 'l'
                classify[training[index, i] == 1, 0]= 'r'
                el= get_entropy_left(classify,labels)
                er= get_entropy_right(classify,labels)
                minInfo=get_info(el,er,classify)
            else:
                for s in  per_split:
                     #find sth percentile and then find the entropy for it.
                     classify=np.chararray((training[index].shape[0],1))
                     labels=training[index,13]
                     p= np.percentile(training[index,i], s)
                     classify[training[index, i] < p, 0] = 'l'
                     classify[training[index, i] >= p, 0]= 'r'
                     
                     el= get_entropy_left(classify,labels)
                     er= get_entropy_right(classify,labels)
                     s_info=get_info(el,er,classify)
                     if(s_info<minInfo):
                         minInfo= s_info
                         minS= p
            split[i]=minS
            info[i]=minInfo
        else:
            split[i]=minS
            info[i]=minInfo
    sIndex=np.argmin(info)
    return sIndex,split[sIndex]


def get_info(el,er,classify):
    c0=classify[classify[:,0] =='l'].shape[0]
    c1=classify[classify[:,0] =='r'].shape[0]
    size= classify.shape[0]
    info= (c0*el/float(size)) + (c1*er/float(size))
    return info
    
def get_entropy_left(classify,labels):
    e=0    
    size= classify[classify[:,0] =='l'].shape[0] 
    if size !=0:
        c10=classify[np.where((classify[:,0] =='l' ) * (labels[:] == 0))].shape[0]
        c11=classify[np.where((classify[:,0] =='l' ) * (labels[:] ==1))].shape[0]
        if c10 ==0 or c11 ==0:
            return 0
        p10= c10/float(size)
        p11= c11/float(size)
        e= (-1*(p10 * log(p10,2))) - (p11 * log(p11,2))
    return e
    
def get_entropy_right(classify,labels):
    e=0
    size= classify[classify[:,0] =='r'].shape[0]
    if size !=0:
        c10=classify[np.where((classify[:,0] =='r' ) * (labels[:] == 0))].shape[0]
        c11=classify[np.where((classify[:,0] =='r' ) * (labels[:] == 1))].shape[0]
        if c10 ==0 or c11 ==0:
            return 0
        p10= c10/float(size)
        p11= c11/float(size)
        e= (-1*(p10 * log(p10,2))) - (p11 * log(p11,2))
    return e
        

def classify(feat_split,bin_label,X,yPred,b):

    pred= np.zeros((yPred.shape[0],1))
    for x in range(X.shape[0]):
       if X[x,int(feat_split[0,0])]<= feat_split[0,1]:
           if feat_split[1,0]==-1:
               pred[x,0]=np.random.choice(X[:,13], 1)
           else:
                if X[x,int(feat_split[1,0])]<= feat_split[1,1]:
                    pred[x,0]= bin_label[0]
                else:
                    pred[x,0]= bin_label[1]
       else:
            if feat_split[2,0]==-1:
               pred[x,0]=np.random.choice(X[:,13], 1)
            else:
                if X[x,int(feat_split[2,0])]<= feat_split[2,1]:
                    pred[x,0]= bin_label[2]
                else:
                    pred[x,0]= bin_label[3]    
    
    yPred[:,b]= pred[:,0]   
    return yPred

def cal_error(yP,d,alpha):
    yP[yP==0]=-1
    #yP[yP==0]=1
    yP= np.multiply(alpha,yP)
    predict= np.sum(yP, axis=1)
    predict[predict <= 0]=0
    predict[predict > 0]=1
    error=np.sum(np.abs(np.subtract(predict,d[:,13])))
    error_rate= error*100/float(d.shape[0])
    return error_rate

def classifier(data,num_crossval,B):
    train_error=[]
    test_error=[]
    foldsize= len(data)/num_crossval
    for k in range(num_crossval):
        testing = data[k*foldsize:(k+1)*foldsize]
        a=data[:k*foldsize]
        b= data[(k+1)*foldsize:]
        training = np.concatenate(( a , b))
        
        N= training.shape[0]
        y=np.zeros((N,1))
        y[:,0]= training[:,13]
        w = np.ones((N,1))/float(N)        
        yPred= np.zeros((N,B))
        yPredTest= np.zeros((testing.shape[0],B))
        alpha= np.zeros((B))
        b=0
        #adaboost for B classifiers
        while b < B:
            #ADABOOST ALGO       
            #create training set with sampling according to weight
            idx= np.random.choice(N, N, p=w[:,0])
            #base classifier
            feat_split,bin_label=create_tree(idx,training)
            
            yPred= classify(feat_split,bin_label,training,yPred,b)
            yPredTest= classify(feat_split,bin_label,testing,yPredTest,b)
            #use training to get yPred Nx1 y label for classifying all data
            error= np.abs(np.subtract(yPred[:,b],y[:,0]))
            et = np.dot(w.T, error)[0]
           # et= np.sum(et)
 #           print(et)
            if(et>=0.5 or et<=0.00001):
                #print(et)
                w = np.ones((N,1))/float(N)
                continue
            #error=error.T
            error[error[:]==1]=1
            error[error[:]==0]=-1
            alpha[b]= 0.5*np.log((1-et)/et)
            error= np.exp(error*alpha[b])
            w=  np.multiply(error,w.T) 
            w=w/np.sum(w)
            w=w.T
            b +=1
        train=cal_error(yPred,training,alpha)
        test=cal_error(yPredTest,testing,alpha)
        train_error.append(train)
        test_error.append(test)
        #print(alpha)
    return train_error, test_error

          
filename = sys.argv[1]
BValue= sys.argv[2].split(",")
Bs= map(int,BValue)
num_crossval= int(sys.argv[3])


boston50 = np.genfromtxt(filename , delimiter=',')
boston75 = np.genfromtxt(filename , delimiter=',')


median= np.percentile(boston50[:,13], 50)
thirdquart= np.percentile(boston75[:,13], 75)

boston50[boston50[:, 13] < median, 13] = 0
boston50[boston50[:, 13] >= median, 13] = 1

boston75[boston75[:, 13] < thirdquart, 13] = 0
boston75[boston75[:, 13] >= thirdquart, 13] = 1
b50train=[]
b50test=[]
print('For Boston 50 :')
for B in Bs:
    print 'for B=' , B
    traine50,teste50= classifier(boston50,num_crossval,B)
    print'k fold train error:',traine50
    print 'k fold test error:',teste50
    sdTrain=np.std(traine50, axis=0)
    train=np.sum(traine50, axis=0)/float(num_crossval)
    sdTest=np.std(teste50, axis=0)
    test=np.sum(teste50, axis=0)/float(num_crossval)
    b50train.append(train)
    b50test.append(test)
    print'average train error:',train
    print 'average test error:',test
    print 'train sd:',sdTrain
    print'test sd:',sdTest

b75train=[]
b75test=[]
print('For Boston 75 :')
for B in Bs:
    print 'for B=' , B
    traine75,teste75= classifier(boston75,num_crossval,B)
    print'k fold train error:',traine75
    print 'k fold test error:',teste75
    sdTrain=np.std(traine75, axis=0)
    train=np.sum(traine75, axis=0)/float(num_crossval)
    sdTest=np.std(teste75, axis=0)
    test=np.sum(teste75, axis=0)/float(num_crossval)
    b75train.append(train)
    b75test.append(test)
    print'average train error:',train
    print 'average test error:',test
    print 'train sd:',sdTrain
    print'test sd:',sdTest













