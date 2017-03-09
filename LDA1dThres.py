# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import sys 

def classifier(data,num_crossval):
    train_error=[]
    test_error=[]
    foldsize= len(data)/num_crossval
    for k in range(10):
        testing = data[k*foldsize:(k+1)*foldsize]
        a=data[:k*foldsize]
        b= data[(k+1)*foldsize:]
        training = np.concatenate(( a , b))
    
        #apply lda on these
        x0= training[training[:, 13]==0]
        x0= x0[:,0:-1]
        
        x1= training[training[:, 13]==1]
        x1= x1[:,0:-1]
        
        m1=x1.mean(axis=0)
        m1=m1.reshape(13,1)
        m1=np.transpose(m1)
        m0=x0.mean(axis=0)
        m0=m0.reshape(13,1)
        m0=np.transpose(m0)
         
        sb= (m1-m0)*(m1-m0).T
        
        tempsw1=0;
        tempsw0=0
        for i in range(len(x1)):
            diff= x1[i,:]-m1
            tempsw1= tempsw1 + diff*diff.T
        for i in range(len(x0)):
            diff= x0[i,:]-m0
            tempsw0= tempsw0 + diff*diff.T
        
        sw= tempsw0+tempsw1
        d=m1-m0
        d=d.T
        swInv= np.linalg.inv(sw)
        w= np.dot(swInv,d)
        
        thresh= np.dot((m1+m0)/2,w)
        incorrect=0
        for i in range(len(testing)):
            label=0
            fx= np.dot(testing[i,0:-1],w)
            if fx > thresh:
                label=1
            if label !=testing[i,13]:
                incorrect= incorrect+1
        #print(incorrect,len(testing) )
        errorrate= incorrect/float(len(testing))
        test_error.append(errorrate)
        incorrect=0
        for i in range(len(training)):
            label=0
            fx= np.dot(training[i,0:-1],w)
            if fx > thresh:
                label=1
            if label !=training[i,13]:
                incorrect= incorrect+1
       # print(incorrect,len(training) )
        e= incorrect/float(len(training))
        train_error.append(e)
        
    avg_test_error= np.sum(test_error)/num_crossval
    avg_train_error= np.sum(train_error)/num_crossval
    testSd= np.sqrt(np.sum((x- avg_test_error)**2 for x in test_error)/float(10))
    trainSd=np.sqrt(np.sum((x- avg_train_error)**2 for x in train_error)/float(10))
    print('train standard deviation:', trainSd , 'test standard deviation: ', testSd)
    return avg_test_error, avg_train_error

          
filename = sys.argv[1]
num_crossval= int(sys.argv[2])


boston50 = np.genfromtxt(filename , delimiter=',')
boston75 = np.genfromtxt(filename , delimiter=',')


median= np.percentile(boston50[:,13], 50)
thirdquart= np.percentile(boston75[:,13], 75)

boston50[boston50[:, 13] < median, 13] = 0
boston50[boston50[:, 13] >= median, 13] = 1

boston75[boston75[:, 13] < thirdquart, 13] = 0
boston75[boston75[:, 13] >= thirdquart, 13] = 1
print('For Boston 50 :')
teste50, traine50= classifier(boston50,num_crossval)

print("test error percent for boston50 is: ", teste50*100, ' and train error is: ',traine50*100 )
print('For Boston 75 :')
teste75, traine75= classifier(boston75,num_crossval)
print("test error percent for boston75 is: ", teste75*100, ' and train error is: ',traine75*100 )












