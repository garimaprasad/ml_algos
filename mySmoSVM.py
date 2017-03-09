# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 12:26:22 2016

@author: Garima
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

#############################################
def calculatePrimal(alpha):
    return 0.5 * np.dot((np.dot(alpha.T, Q)), alpha) + np.dot(p.T, alpha)

#############################################
def calculateGradient(alpha):
    f = np.dot(Q, alpha) + p
    return f

#############################################
def getI(alpha):
    Iup = []
    Ilow = []
    for t in range(0, len(alpha)):
        if (alpha[t, 0] < C and y[t, 0] == 1) or (alpha[t, 0] > 0 and y[t, 0] == -1):
            Iup.append(t)
        elif (alpha[t, 0] < C and y[t, 0] == -1) or (alpha[t, 0] > 0 and y[t, 0] == 1):
            Ilow.append(t)
    return Iup, Ilow

#############################################
def getWSS(alpha, f):
    B = []
    Iup, Ilow = getI(alpha)
    bts = np.zeros((l))
    IMax = -1 * np.Inf
    ii = 0
    for tt in Iup:
        v = -1 * y[tt, 0] * f[tt, 0]
        if v > IMax:
            IMax = v
            ii = tt

    temp = -1 * y[ii, 0] * f[ii, 0]
    for s in range(0, l):
        bts[s] = -1 * y[ii, 0] * f[ii, 0] + y[s, 0] * f[s, 0]

    jMin = np.Inf
    jj = 0
    for ss in Ilow:
        if -1 * y[ss, 0] * f[ss, 0] < temp:
            val = np.power(bts[ss], 2) / (-1 * ats[ii, ss])
            if val < jMin:
                jMin = val
                jj = ss
    B.append(ii)
    B.append(jj)
    return B

#############################################
def smo(alpha):
    k = 1
    e = 0.0001
    fcur = fprev = calculatePrimal(alpha)
    farr = []
    farr.append(fprev[0])
    flag = True
    while flag and k <= 1000:
        f = calculateGradient(alpha)
        fprev = fcur
        B = getWSS(alpha, f)
        i = B[0]
        j = B[1]
        # update alpha        
        if y[i, 0] != y[j, 0]:
            delta = ((-1 * f[i, 0]) + (-1 * f[j, 0])) / ats[i, j]
            diff = alpha[i, 0] - alpha[j, 0]
            alpha[i, 0] += delta
            alpha[j, 0] += delta
            if diff > 0:
                if alpha[j, 0] < 0:
                    alpha[j, 0] = 0
                    alpha[i, 0] = diff
                if alpha[i, 0] > C:
                    alpha[j, 0] = C - diff
                    alpha[i, 0] = C
            else:
                if alpha[i, 0] < 0:
                    alpha[i, 0] = 0
                    alpha[j, 0] = -diff
                if alpha[j, 0] > C:
                    alpha[j, 0] = C
                    alpha[i, 0] = C + diff
        if y[i, 0] == y[j, 0]:
            delta = ((-1 * f[i, 0]) + f[j, 0]) / ats[i, j]
            summ = alpha[i, 0] + alpha[j, 0]
            alpha[i, 0] += delta
            alpha[j, 0] -= delta

            if summ <= C:
                if alpha[j, 0] < 0:
                    alpha[j, 0] = 0
                    alpha[i, 0] = summ
                if alpha[i, 0] < 0:
                    alpha[j, 0] = summ
                    alpha[i, 0] = 0
            elif summ >= C:
                if alpha[j, 0] > C:
                    alpha[i, 0] = summ - C
                    alpha[j, 0] = C
                if alpha[i, 0] > C:
                    alpha[j, 0] = summ - C
                    alpha[i, 0] = C
        fcur = calculatePrimal(alpha)
        if k > 1 and fprev - fcur <= e:
            flag = False
        farr.append(fcur[0])
        k += 1
    return k, farr
    
##########################################
   
filename= sys.argv[1]
runs= int(sys.argv[2])
runs=5
data = np.genfromtxt(filename, delimiter=',')
x = data[:, 1:]
x = x / np.linalg.norm(x, axis = 1)[:,None]
y = data[:, 0]
y[y[:] == 3] = -1
y = y.reshape(2000, 1)
l = x.shape[0]
yx = np.multiply(y, x)
Q = np.dot(yx, yx.T)
K = np.dot(x, x.T).astype('float')
C = 0.05
runs=5
times=[]
ats = np.zeros((l, l))
tao = 0.0001


#############################################
for m in range(0, l):
    for n in range(0, l):
        ats[m, n] = K[m, m] + K[n, n] - 2 * K[m, n]
        if ats[m, n] <= 0:
            ats[m, n] = tao
#############################################           
for run in range(runs):
    
    a = np.zeros((l, 1))
    p = -1 * np.ones((l, 1))
    start= time.time()
    k, farr = smo(a)
    end= time.time()
    times.append(end-start)
    print(end-start)  
meanTime = np.mean(times, axis=0)
sd=np.std(times, axis=0)
print 'average time' , meanTime
print 'standard deviation', sd
    

