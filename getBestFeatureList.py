import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

error_m=10000
final_ar=[]

def getErrorM():
    return error_m

def setErrorM(tmp):
    global error_m
    error_m=tmp

def getError(cols_to_use):
    df_x = pd.read_csv('processed.cleveland.data.csv',usecols=cols_to_use,header=None)
    df_y = pd.read_csv('processed.cleveland.data.csv',usecols=[56],header=None)

    x=df_x.values.tolist()
    y=df_y.values.tolist()


    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(500,), random_state=1)
    clf.fit(x,y)

    test_case=x[-50:]
    answers=y[-50:]

    predicted=clf.predict(test_case)

    error=0
    i=0
    while i<len(answers):
        if answers[i]!=predicted[i]:
            error=error+1
        i=i+1

    return(error)


def printCombination(arr,n,r):
    data=[0]*10
    combinationUtil(arr, n, r, 0, data, 0);

def combinationUtil(arr, n, r, index, data, i):
    if index==r:
        tmp=[]
        for j in range(0,r):
            tmp.append(data[j])
        tmp_e=getError(tmp)
        if(tmp_e<getErrorM()):
            setErrorM(tmp_e)
            final_ar=tmp
            print("got a better ar %s with error %s and global error %s"%(final_ar,tmp_e,getErrorM()))
        return 
    if i>=n:
        return
    data[index] = arr[i]
    combinationUtil(arr, n, r, index+1, data, i+1)
    combinationUtil(arr, n, r, index, data, i+1)
    

def colGen():
    cols_to_use=[2,3,8,9,11,15,18,31,37,39,40,43,50]
    for r in range(1,len(cols_to_use)):
        print("Executing r = %s"%(r))
        start_time_1 = time.time()
        printCombination(cols_to_use, len(cols_to_use), r)
        print("--- %s seconds of case %s ---" % (time.time() - start_time_1,r))
            
            
    
start_time = time.time()    
colGen()
print("--- %s seconds ---" % (time.time() - start_time))