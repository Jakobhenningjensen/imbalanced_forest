#%%
from impurities import gini,entropy
class RegTree:
    def __init__(self,criterion = "gini"):
        print("Hello RegTree!")
        self.node = self.NODE()
        if criterion=="gini":
            self.impurity= gini
        elif criterion=="entropy":
            self.impurity = entropy
        else:
            raise ValueError("Criterion can either be 'gini' or 'entropy'")

    def test_split(self,targets,feature,split):
        n_data = len(targets) #Number of data points
        idx_greater = [p[0] for p in np.argwhere(feature>=split)] #idx for greater split
        idx_lower = list(set(range(len(targets)))-set(idx_greater)) #idx for lower split
        imp_greater = self.impurity(targets[idx_greater]) #impurity for greater
        imp_lower = self.impurity(targets[idx_lower]) #impurity lower
        return (idx_greater,idx_lower,len(idx_greater)/n_data*imp_greater+len(idx_lower)/n_data*imp_lower)
            
    def get_split(self,node):
        #Returns the feature/split which maximizes gain
        targets= node.targets
        data = node.data
        feature=0 #feature idx
        split=0.0
        low_imp =np.inf
        idx_greater=0
        idx_lower=0
        for i,f in enumerate(data): #for each feature calculate the midpoint
            splits = (f[1:]+f[:-1])/2 #midpoint for each split
            for s in splits:
                idx_g,idx_l,imp = self.test_split(targets=targets,feature=f,split=s)
                if imp<low_imp: #new best split
                    split = s #Best plit
                    feature=i #new feature index
                    idx_greater=idx_g
                    idx_lower=idx_l
                    low_imp=imp
        return (idx_greater,idx_lower,feature,split)
        

         
    def run(self,node):        
        #Splits the node in children nodes untill pure nodes
        pure_nodes = (len(np.unique(node.targets))==1)
        while not pure_nodes: #not pure node
            idx_greater,idx_lower,_,_=self.get_split(node)
            node.greater(self.NODE(data=node.data[idx_greater],targets=node.targets[idx_greater]))
            node.lower(self.NODE(data=node.data[idx_lower],targets=node.targets[idx_lower]))
        self.run(node.greater)
        self.run(node.lower)           


        

    class NODE:
        def __init__(self,greater=[],lower=[],data=[],targets=[]):
            self.data = data
            self.lower = lower
            self.greater = greater
            self.targets=targets


#%%
from sklearn.datasets import load_breast_cancer as load_data
import numpy as np

DATA = load_data()
X= DATA.data
y = DATA.target

regtree=RegTree(criterion="gini")

node = regtree.NODE(data=np.array([[20,30,40,50],[1,0,0,1]]),targets=np.array([1,1,0,0]))
regtree.run(node)


node = regtree.NODE()

node.data=X.transpose()

for f in node.data:
            temp = pd.DataFrame({"f":f,"t":targets})
            temp.sort("t",d)
            split_points = [] 

        return (feature,split)




regtree = RegTree()
regtree.__private_test()n