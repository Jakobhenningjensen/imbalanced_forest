from sklearn.datasets import load_breast_cancer as load_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DCT
from main import RegTree
import numpy as np
import unittest
import time



class TestPred(unittest.TestCase):

    def test_pred(self):
        DATA = load_data() #Loads test data
        X= DATA.data
        y=DATA.target
        X_train, X_test, y_train, _ = train_test_split(X,y, test_size=0.2, random_state=42)
        
        #max_depth=[np.inf,None]
        max_depth=[10]*2
        regtree=RegTree(criterion="gini",max_depth=max_depth[0])
        regtree.fit(X_train,y_train)
        pred_pred = regtree.predict(X_test)

        true_tree = DCT(min_impurity_split=0,max_depth=max_depth[1]) #Sklearn 
        true_tree.fit(X_train,y_train)
        pred_true = true_tree.predict(X_test)

        pred_true==pred_pred
        self.assertEqual(sum(pred_pred!=pred_true),0, "Not same predictions as sklearn!")


if __name__=="__main__":
    
    unittest.main()

