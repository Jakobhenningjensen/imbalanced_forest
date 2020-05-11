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



    def possible_splits(self,feature,y):
        """
        Returns an index array for values to consider.
        Only consider cases where y[i]!=y[i+1] when sorted according to the given feature
        """

        yi = y[:-1]
        yi1= y[1:]
        idx=np.argwhere((yi1-yi)!=0)
        return idx.flatten()

    

    def test_split(self,X,y,splits):
        """
        Calculates the weighted impurity for each splitting point in splits

        """
        n_data = len(y) #Number of data points
        splits=(X[splits]+X[splits+1])/2

        idx_greater = (X>splits[:,None]) #index for greater split
        idx_lower = (X<splits[:,None]) #index for lower split

        imp_greater =[self.impurity(y[idx]) for idx in idx_greater]  #impurity for greater
        imp_lower = [self.impurity(y[idx]) for idx in idx_lower] #impurity lower

        impur = [sum(idx_great)/n_data*imp_great+sum(idx_low)/n_data*imp_low for idx_great,imp_great,idx_low,imp_low in zip(idx_greater,imp_greater,idx_lower,imp_lower)] #Weighted impurity
        return (impur,splits)

    
    
    def get_split(self,X,y):
        """
         For all features, find the best splitting point in the best feature
        
        """
            
        BEST_COL = 0
        BEST_SPLIT =0
        BEST_IMPUR = 99
        for i,feature in enumerate(X.T):
            arg_sort=np.argsort(feature) #Sort the feature for optimizing the find of splitting points
            feature= feature[arg_sort]
            y_sort = y[arg_sort]
            splits = self.possible_splits(feature,y_sort)  #Get    

            impur,splits = self.test_split(feature,y_sort,splits) #Get impurity for splitting points
            best_idx = np.argmin(impur)
            best_impur = impur[best_idx]
            
            if best_impur==0.0: #Found perfect split, terminate
                return(i,splits[best_idx])
            elif best_impur<BEST_IMPUR:
                BEST_IMPUR=best_impur
                BEST_SPLIT=splits[best_idx]
                BEST_COL=i
        return (BEST_COL,BEST_SPLIT)


    

    def fit(self,X,y,par_node={},depth=0):

        if len(np.unique(y))==1: #Only one class in the node; return that
            
                return ({"class":y[0]})
        elif depth>=self.max_depth: #Reached max-depths

            return {"class":int(np.round(np.mean(y)))}
        else: #Split Tree 
            col,split=self.get_split(X,y)
            idx_left = (X[:,col]<split) #index for lower split
            idx_right = (X[:,col]>=split) #index for lower split
                        
            par_node={"col":col,#Col=index,
            "split":split, #Splitting value
            "class":int(np.round(np.mean(y)))} #Class 
            par_node["left"]=self.fit(X[idx_left,:],y[idx_left],{},depth+1) #Fit left tree
            par_node["right"]=self.fit(X[idx_right,:],y[idx_right],{},depth+1) #Fit right tree

            self.tree = par_node #Save the tree
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
