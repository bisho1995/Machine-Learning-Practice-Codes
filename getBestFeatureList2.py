# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:05:52 2017

@author: Bisvarup Mukherjee
"""

import time
import pandas as pd
from sklearn.neural_network import MLPClassifier




# ****************************** start of class *******************************



class BestFeature:
    
    
    
    
    def __init__(self,filename,arr,target,hidden_layer_size=100,start_from=1):
        self.error_m=10000
        self.final_ar=[]
        self.file_name=filename
        self.target=target
        self.hidden_layer_size=hidden_layer_size
        self.test_data_offset=-50
        self.ar=arr
        self.start_from=start_from
        
        
        
        
        
        
        
    def getError(self,cols_to_use):
        df_x = pd.read_csv(self.file_name,usecols=cols_to_use,header=None)
        df_y = pd.read_csv(self.file_name,usecols=self.target,header=None)
    
        x=df_x.values.tolist()
        y=df_y.values.tolist()
    
    
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(self.hidden_layer_size,), random_state=1)
        clf.fit(x,y)
    
        test_case=x[self.test_data_offset:]
        answers=y[self.test_data_offset:]
    
        predicted=clf.predict(test_case)
    
        error=0
        i=0
        while i<len(answers):
            if answers[i]!=predicted[i]:
                error=error+1
            i=i+1
    
        return(error)

        
        
        
        
        
        
        
        
        
        
        
    def printCombination(self,n,r):
        data=[0]*10
        self.combinationUtil(n, r, 0, data, 0)
            
    
    
    
    
    
    
    
    
    
    def printBestFeatureList(self):
        print("%s with max error %s"%(self.final_ar,self.error_m))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    def combinationUtil(self, n, r, index, data, i):
        if index==r:
            tmp=[]
            for j in range(0,r):
                tmp.append(data[j])
            tmp_e=self.getError(tmp)
            if(tmp_e<self.error_m):
                self.error_m=tmp_e
                self.final_ar=tmp
                print("got a better ar %s with error %s "%(self.final_ar,tmp_e))
            return 
        if i>=n:
            return
        data[index] = self.ar[i]
        self.combinationUtil(n, r, index+1, data, i+1)
        self.combinationUtil(n, r, index, data, i+1)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def main(self):
        start_time=time.time()
        try:
            
            print(" =================  Starting program  =============")
            total_iterations=len(self.ar)
            print("There is going to be %s iterations"%(total_iterations))
            for r in range(self.start_from,len(self.ar)):
                print("Executing iteration %s/%s please wait"%(r,total_iterations))
                start_time_1 = time.time()
                self.printCombination( len(self.ar), r)
                print("--- %s seconds of case %s ---" % (time.time() - start_time_1,r))
        except Exception  as e:
            
            print("Unfortunately an error occured %s",e)
        finally:
            self.printBestFeatureList()
            print("--- %s seconds ---" % (time.time() - start_time))
            print(" =================  Stopping program  =============")
        










# ********************* End of class **********************************



#filename columns to use target hidden layer size
obj1=BestFeature('processed.data.csv',[0,1,2,3,4,5,6,7,8,9,10,11,12],[13],600,6)
obj1.main()
        