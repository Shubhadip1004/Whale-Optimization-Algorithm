import numpy as np
import pandas as pd
import math
import csv
import sys
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

def sigmoid(t, rn):                  # defining the sigmid function f(x): (e ^ ((2 * x) - 1)) / (e ^ ((2 * x) + 1))
    a = np.zeros(d-1)
    for i in range(d-1):
        if rn[i] < (math.exp(2 * abs(t[i]))-1)/(math.exp(2 * abs(t[i]))+1):
            a[i] = 1
        else:
            a[i] = 0
    return a

def init_fitness(sample):                # getting the fitness value of the whole population at once
    fitness = np.zeros([len(sample), 1])
    for i in range(len(sample)):
        row = sample[i, :]
        col = []
        for j in range(41):
            if row[j] == 1.0:
                col.append(str(j))
        stripped_data = np.array(DS[col])
        # splits stripped_data into train_x and test_x , target_data into train_y and test_y in 7:3 ratio
        train_x, test_x, train_y, test_y = train_test_split(stripped_data, target_data, test_size=0.3)
        DCST = DecisionTreeClassifier(max_depth=41, criterion="gini")
        DCST.fit(train_x, train_y.ravel())
        pred_y = DCST.predict(test_x)
        cfm = confusion_matrix(test_y, pred_y)
        fitness[i, 0] = np.trace(cfm) / np.sum(cfm)
    return fitness

def inter_fitness(sample):              # getting the fitness value of one population at a time
    row = sample
    col = []
    for j in range(41):
        if row[j] == 1:
            col.append(str(j))
    stripped_data = np.array(DS[col])
    train_x, test_x, train_y, test_y = train_test_split(
        stripped_data, target_data, test_size=0.3)
    DCST = DecisionTreeClassifier(max_depth=41, criterion="gini")
    DCST.fit(train_x, train_y.ravel())
    pred_y = DCST.predict(test_x)
    cfm = confusion_matrix(test_y, pred_y)
    fitness = np.trace(cfm) / np.sum(cfm)
    return fitness

def dot(a,b):
    s = 0
    v = (a.shape == b.shape)
    if v == True:
        for i in range(a.size):
            s += a[i] * b[i]
        return s
    else:
        print("Shape is not equal")
        print("Shape of A is:",a.shape)
        print("Shape of B is:",b.shape)
        sys.exit()

def updation_1(ppl, best_whale,vec_A,vec_C,i):              # value updation process 1
    y= np.zeros(d-1)
    D = abs(dot(best_whale,vec_C) - ppl[i,:])
    for j in range(d-1):
        ppl[i,j] = best_whale[j]+ dot(D, vec_A)
    y = ppl[i,:]
    x = np.count_nonzero(y == 0.0)
    if x < 4:
        y = np.random.choice(a=[1.0, 0.0], size=((d-1)), p=[0.8, 0.2])
    ppl[i,:] = sigmoid(y,np.random.rand(d-1))
    return ppl

def updation_2(ppl, best_whale,vec_A,vec_C,i):              # value updation process 2
    y= np.zeros(d-1)
    r = random.randint(0,n-1)
    D = abs(np.dot(ppl[r,:],vec_C) - ppl[i,:])
    for j in range(d-1):
        ppl[i,j] = ppl[r,j]+ dot(D, vec_A)
    y = ppl[i,:]
    x = np.count_nonzero(y == 0.0)
    if x < 4:
        y = np.random.choice(a=[1.0, 0.0], size=((d-1)), p=[0.8, 0.2])
    ppl[i,:] = sigmoid(y,np.random.rand(d-1))
    return ppl

def updation_3(ppl, best_whale,i,b,l):              # value updation process 3
    y= np.zeros(d-1)
    dst = np.linalg.norm(best_whale -ppl[i,:])
    for j in range(d-1):
        ppl[i,j] = best_whale[j] + dst * (np.e ** (b*l)) * math.cos(2 * math.pi * l) 
    y = ppl[i,:]
    x = np.count_nonzero(y == 0.0)
    if x < 4:
        y = np.random.choice(a=[1.0, 0.0], size=((d-1)), p=[0.8, 0.2])
    ppl[i,:] = sigmoid(y,np.random.rand(d-1))
    return ppl

def WOA(ppl, fit, bestitr):                          # Whale Optimisation Algorithm
    best_whale = np.zeros((1,d-1))
    b_whale = np.hstack([ppl,fit])                       
    b_w = b_whale[np.argsort(b_whale[:,-1])[::-1]]
    best_whale = np.array(b_w [0,:d-1])
    a = np.linspace(2,0,itr)
    b = 1.5     # logarithmic constant
    for it in range(itr):
        r = np.random.random()
        b_whale = np.hstack([ppl,fit])
        b_w = b_whale[np.argsort(b_whale[:,-1])[::-1]]
        best_whale = np.array(b_w [0,:d-1])
        vec_A = []
        vec_C = []
        for w in range(d-1):
            vec_A .append(2 * a[it] * r - a[it])
            vec_C .append(2 * r)
        vec_A = np.array(vec_A)
        vec_C = np.array(vec_C)
        for i in range(n):
            p = random.random()
            l = random.uniform(-1.0,1.0)
            if (p<0.5):
                if (np.linalg.norm(vec_A[i]-np.zeros(vec_A[i].shape))) < 1:
                    # print("updation 1")
                    ppl = updation_1(ppl,best_whale,vec_A,vec_C,i)
                    fit = init_fitness(ppl)
                else:
                    # print("updation 2")
                    ppl = updation_2(ppl,best_whale,vec_A,vec_C,i)
                    fit = init_fitness(ppl)
            else:
                # print("updation 3")
                ppl = updation_3(ppl,best_whale,i,b,l)
                fit = init_fitness(ppl)
            b_whale = np.hstack([ppl,fit])                       
            b_w = b_whale[np.argsort(b_whale[:,-1])[::-1]]
            best_whale = np.array(b_w [0,:d-1])
        b_whale = np.hstack([ppl,fit])
        b_w = b_whale[np.argsort(b_whale[:,-1])[::-1]]
        bs = np.array(b_w [0,:])
        ylst.append(bs[-1:])
        xyz = 42 - np.count_nonzero(bs)
        print("\nBest of Iteration ::: ", it+1, "Fetaures deleted ::: ",xyz," Fetaures selected ::: ",d-1-xyz,":\n", bs)
        bestitr[it] = np.array(bs)
        with open(name, 'a', newline='\n') as out:
            writer = csv.writer(out, delimiter=',')
            rsl = np.concatenate([["Iter "+str(it+1)], bs])
            writer.writerow(rsl)
        out.close()

print("\nProcess Started")
d = 42
n = 30
itr = 100
xlst = [i for i in range(1,itr+1)]
ylst = []
# probability of number of 1's in randomly generated 'ppl'
p = 0.7
DS = pd.read_csv(r"C:\Users\shubh\Desktop\Machine Learning\Database Analysis\train_nslkdd_2class_preprocessed_normalized.csv")    # file name is changed 
data = np.array(DS)
# deletes d-1 th stream along axis 1 (columns) from 'data' and store into 'training_data'
training_data = np.delete(data, d-1, 1)
# deletes first d-1 streams along axis 1 (columns) from 'data' and store into 'target_data'
target_data = np.delete(data, np.s_[:d-1], 1)
name = "NSLKDD2classWOAresult.csv"
# generates 'ppl' having n*(d-1) sized array containg 1's and 0's
ppl = np.random.choice(a=[1.0, 0.0], size=(n, d-1), p=[p, 1-p])
# calculates fitness of each row in 'ppl' and store into 'fit'
fit = init_fitness(ppl)
# merges 'ppl' and its 'fit' and store into 'arr'
arr = np.hstack([ppl, fit])
# 'np.argsort' gives indexes of sorted 'arr' , reverse into decreasing order , store into 'rec'
rec = arr[np.argsort(arr[:, -1])]
# stores each stream excluding stream's last element (fit) and rewrite into 'ppl'
ppl = np.array(rec[:, :-1])
# stores each stream's last element (fit) and rewrite into 'fit'
fit = np.array(rec[:, -1:])
bestitr = np.zeros([itr, d])
toprow = []
for l in range(0, d-1):
    toprow.append("Dim "+str(l+1))
with open(name, 'w', newline='\n') as op:
    writer = csv.writer(op, delimiter=',')
    header = np.concatenate([["Iter no."], toprow, ["Fitness"]])
    writer.writerow(header)
op.close()
WOA(ppl, fit, bestitr)
final = bestitr[np.argsort(bestitr[:, -1])[::-1]]
print("\nGlobal Best:\n", final[0])
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.title("Whale Optimisation Algorithm")
plt.plot(xlst,ylst)
plt.show()
print("\nProcess Finished\n")