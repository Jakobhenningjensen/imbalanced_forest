#%%
from impurities import gini,entropy
import numpy as np
class RegTree:
    def __init__(self,criterion = "gini",max_depth=1):
        self.max_depth=max_depth
        self.tree={}
        if criterion=="gini":
            self.impurity= gini
        elif criterion=="entropy":
            self.impurity = entropy
        else:
            raise ValueError("Criterion can either be 'gini' or 'entropy'")



    def possible_splits(self,targets):
        """
        Returns an index array for values to consider.
        Only consider cases where y[i]!=y[i+1]
        """
        yi = targets[:-1]
        yi1= targets[1:]
        idx=np.argwhere((yi1-yi)!=0)
        return idx.flatten()+1

    

    def test_split(self,X,targets,split):
        n_data = len(targets) #Number of data points
        idx_greater = [p[0] for p in np.argwhere(X>=split)] #index for greater split
        idx_lower = list(set(range(len(targets)))-set(idx_greater)) #index for lower split
        imp_greater = self.impurity(targets[idx_greater]) #impurity for greater
        imp_lower = self.impurity(targets[idx_lower]) #impurity lower
        impur = len(idx_greater)/n_data*imp_greater+len(idx_lower)/n_data*imp_lower #Weighted impurity
        return impur
    
    
    def get_split(self,X,y):
        """ For all columns, find the best splitting point in the best column
        """
        BEST_IMPUR = 10.0
        BEST_COL=0
        BEST_SPLIT =0
        for i,feature in enumerate(X.T): #For all features      

            possible_splits = self.possible_splits(feature) #Look at all possible splits
            possible_splits=feature[possible_splits]
            for split in possible_splits:
                impur = self.test_split(feature,y,split)
                if impur<BEST_IMPUR: #And save the best

                    BEST_IMPUR=impur
                    BEST_SPLIT=split
                    BEST_COL = i
        return(BEST_COL,BEST_SPLIT)
    
    

    def fit(self,X,y,par_node={},depth=0):

        if len(y)==0: #No data in this node

            return None
        elif len(np.unique(y))==1: #Only one class in the node; return that
            
                return ({"class":y[0]})
        elif depth>=self.max_depth: #Reached max-depth

            return {"class":int(np.round(np.mean(y)))}
        else: #Split Tree
            col,split=self.get_split(X,y)
            idx_left = (X[:,col]<split) #index for lower split
            idx_right = (X[:,col]>=split) #index for lower split
                        
            par_node={"col":col,#Col=index,
            "split":split, #Splitting value
            "class":int(np.round(np.mean(y)))} #Class 
            par_node["left"]=self.fit(X[idx_left,:],y[idx_left],{},depth+1)
            par_node["right"]=self.fit(X[idx_right,:],y[idx_right],{},depth+1)

            self.tree = par_node
            return (par_node)
    
    
    def __get_prediction__(self,row):
        "Prediction of an instance"

        tree = self.tree

        while tree.get("split"): #If not leaf node
            if row[tree.get("col")]<tree.get("split"):
                tree = tree.get("left")
              
            elif row[tree.get("col")]>=tree.get("split"):
                tree=tree.get("right")
              
        else:
            return (tree.get("class")) #Leaf-node = return class
    def predict(self,X):
        """
        function for predicting 
        """
        results = np.array([self.__get_prediction__(row) for row in X])
        return results


#%%
from sklearn.datasets import make_classification as mc
import matplotlib.pyplot as plt

X,y = mc(n_samples=100,n_features=2,class_sep=5.0,n_redundant=0)
X_lower_left = (X[:,0]<0) & (X[:,1]<0)

y[X_lower_left]=1-y[X_lower_left]

plt.scatter([p[0] for p in X],[p[1] for p in X],c=y)

regtree=RegTree(criterion="gini",max_depth=10)
regtree.fit(X,y)
pred = regtree.predict(X)
print(sum(y==pred)/len(y))


#%%

from sklearn.datasets import load_breast_cancer as load_data
from sklearn.model_selection import train_test_split
import time


DATA = load_data()
X= DATA.data
y=DATA.target

t = time.time()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
regtree=RegTree(criterion="gini",max_depth=10)
regtree.fit(X_train,y_train)
print(time.time()-t)


pred = regtree.predict(X_test)
print(sum((y_test==pred)/len(y_test)))



#%%


