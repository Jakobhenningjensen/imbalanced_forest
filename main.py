#%%
from impurities import gini,entropy
class RegTree:
    def __init__(self,criterion = "gini",max_depth=1):
        self.max_depth=max_depth
        if criterion=="gini":
            self.impurity= gini
        elif criterion=="entropy":
            self.impurity = entropy
        else:
            raise ValueError("Criterion can either be 'gini' or 'entropy'")

    def test_split(self,feature,targets,split):
        n_data = len(targets) #Number of data points
        idx_greater = [p[0] for p in np.argwhere(feature>=split)] #index for greater split
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
        for i,feature in enumerate(X.T): #For all features      
            possible_splits = np.unique(feature) #Look at all possible splits
            for split in possible_splits:
                impur = self.test_split(X,y,split)
                if impur<BEST_IMPUR: #And save the best

                    BEST_IMPUR=impur
                    BEST_SPLIT=split
                    BEST_COL = i
        return(BEST_COL,BEST_SPLIT)
    
    

    def fit(self,X,y,par_node={},depth=0):
        if par_node is None: #Terminated at last tree

            return None
        elif len(y)==0: #No data in this node

            return None
        elif len(np.unique(y))==1: #Only one class in the node; return that

            return ({"val":y[0]})
        elif depth>=self.max_depth:

            return None
        else: #Split Tree
            col,split=self.get_split(X,y)
            idx_left = (X[:,col]<split) #index for greater split
            idx_right = (X[:,col]>=split)
                        
            par_node={"col":col,"split":split,"class":int(np.round(np.mean(y)))}
            par_node["left"]=self.fit(X[idx_left,:],y[idx_left],{},depth+1)
            par_node["right"]=self.fit(X[idx_right,:],y[idx_right],{},depth+1)

            self.trees = par_node
            return (par_node)
    
    
    def __get_prediction__(self,row):
        "Prediction of an instance"

        tree = self.trees
        while tree.get("split"): #If not leaf node
            if row[tree.get("col")]<tree.get("split"):
                tree = tree.get("left")
            else:
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
from sklearn.datasets import load_breast_cancer as load_data
import numpy as np

DATA = load_data()
X_train= DATA.data[:-10]
y_train = DATA.target[:-10]

X_val = DATA.data[-10:]
y_val = DATA.target[-10:]


regtree=RegTree(criterion="gini",max_depth=10)
regtree.fit(X_train,y_train)
pred = regtree.predict(X_val)
print(np.sum((y_val==pred)/len(y_val)))

#%%