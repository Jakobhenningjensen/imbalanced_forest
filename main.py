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

    
    """
    def test_split(self,X,y,splits):
        n_data = len(y) #Number of data points
        splits=X[splits,:]

        idx_greater = (X>=splits[:,None]) #index for greater split
        idx_lower = (X<splits[:,None]) #index for lower split

        imp_greater =[self.impurity(y[idx]) for idx in idx_greater]  #impurity for greater
        imp_lower = [self.impurity(y[idx]) for idx in idx_lower] #impurity lower

        impur = len(idx_greater)/n_data*imp_greater+len(idx_lower)/n_data*imp_lower #Weighted impurity
        return impur
        """
    
    
    def get_split(self,X,y):
        """ For all features, find the best splitting point in the best feature
        
        Returns:
            impur: (n_feature,2):
                Matrix consisting of the (index, split_value)
        """
        n_data = len(y) #Number of data points
        splits = self.possible_splits(y) #Get the splitting index's
        
        splits = X[splits,:] #Getting splitting values
        split_list=splits
        
        splits = splits[:,None]

        #For each feature, calculate the greater/lower points for each splitting criteria
        
        #idx_* output shape (i,j,k): X[i]>=split(j) (feature k)
        #i.e each row, i, corresponds to X[i]>split[:]   (for feature k)
        idx_greater = X>=splits 
        idx_lower = X<splits 
        impur = np.zeros(shape=(X.shape[1],2))
        for i in range(X.shape[1]):
            idx_low = idx_lower[...,i]
            idx_great = idx_greater[...,i]
            
            imp_greater = [self.impurity(y[p]) for p in idx_great]
            imp_lower = [self.impurity(y[p]) for p in idx_low]

            weighted_imp= np.array([sum(y[p_great])/n_data*imp_greater[j]+sum(y[p_low])/n_data*imp_lower[j] for j,(p_great,p_low) in enumerate(zip(idx_great,idx_low))])
            split_idx = np.argmin(weighted_imp) #Get the split index with the lowest impurity for the current feature
            impur[i]=(split_idx,split_list[split_idx,i])

        col = np.argmin(impur,axis=0)[0] 
        split= impur[col,1]
        return col,split  
    

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
            return par_node
    
    
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

if __name__=="__main__":
    RegTree()


    
#%%
"""
from sklearn.datasets import make_classification as mc
import matplotlib.pyplot as plt


X,y = mc(n_samples=100,n_features=2,class_sep=5.0,n_redundant=0)
X_lower_left = (X[:,0]<0) & (X[:,1]<0)

y[X_lower_left]=1-y[X_lower_left]

plt.scatter([p[0] for p in X],[p[1] for p in X],c=y)

regtree=RegTree(criterion="gini",max_depth=10)
regtree.fit(X,y)
pred = regtree.predict(X)
y==pred)/len(y)) 
"""

#%%
from sklearn.tree import DecisionTreeClassifier as DCT
from sklearn import tree 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer as load_data
DATA = load_data() #Loads test data
X= DATA.data
y=DATA.target
X_train, X_test, y_train, _ = train_test_split(X,y, test_size=0.2, random_state=42)

regtree=RegTree(criterion="gini",max_depth=3)
regtree.fit(X_train,y_train)
col,split = regtree.get_split(X_train,y_train)
pred_pred = regtree.predict(X_test)

true_tree = DCT(min_impurity_split=0,max_depth=3) #Sklearn 
true_tree.fit(X_train,y_train)
pred_true = true_tree.predict(X_test)
tree.plot_tree(true_tree,class_names=True)

sum(pred_true!=pred_pred)

#%%
