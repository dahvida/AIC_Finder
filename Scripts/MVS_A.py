import lightgbm as lgb
import numpy as np
from sklearn.metrics import *

class sample_analysis:
    
    def __init__(self,
                 x,
                 y,
                 params = None,
                 verbose = True,
                 seed = 0
                 ):

        #store relevant attributes
        self.x = np.array(x)
        self.y = np.array(y)
        self.dataset = lgb.Dataset(x, y)
        self.verbose = verbose
        self.seed = seed
        
        #load custom param dict or create one with defaults
        if params is dict:
            params["objective"] = "binary"
            self.params = params
            
        elif params == "default":
            self.params = {"reg_lambda": 1.0,               #helps with MVS estimation
                           "objective": "cross_entropy",
                           "verbosity": -1000,
                           "feature_fraction": 0.95,        #needed to inject randomness in training
                           "random_seed": seed
                           }
                
        
    def get_model(self):
        #train model with LightGBM API
        self.model = lgb.train(params = self.params,
                                   train_set = self.dataset
                                   )

        #save number of leaves for each tree
        self.n_leaves = np.max(self.model.predict(self.x, pred_leaf=True),
                                   axis=0)

        #save number of estimators
        self.n_estimators = self.model.num_trees()
        
        #optionally log info
        if self.verbose is True:
            print("[MVS-SA]: LightGBM model constructed, used", self.n_estimators, "trees")    
    
    def get_l1(self, tree_n):
        #get n_leaves for n-th tree
        n = self.n_leaves[tree_n]
        l1 = 0

        #calculate L1 norm for n-th tree
        for i in range(n):
            l1 += np.abs(self.model.get_leaf_output(tree_n, i))

        return l1
        
    def get_score(self, preds, l1):
        #get squared gradient for i-th prediction
        grad = np.subtract(preds, self.y)
        grad_2 = np.multiply(grad, grad)
        
        #get squared hessian with L1 normalization for i-th prediction
        diff = np.subtract(1, preds)
        hessian = np.multiply(preds, diff)
        hessian_2 = np.multiply(hessian, hessian)
        hessian_2 = np.multiply(hessian_2, l1)
        
        #calculate MVS score and force it to be <1
        score = np.sqrt(grad_2 + hessian_2)
        score[score > 1.0] = 1.0
        
        return score
    
    def get_importance(self, return_trajectory = False):
        #prealloc results box with correct size (n_compounds, n_trees)
        vals_box = np.empty((self.x.shape[0], self.n_estimators), dtype=np.float64)

        #loop over all trees
        for i in range(0, self.n_estimators, 1):
            #get preds of i-th LightGBM iteration
            preds = self.model.predict(self.x,
                                  start_iteration=0,
                                  num_iteration=i)
            
            #get L1 norm of leaf weights for i-th tree 
            l1 = self.get_l1(i)

            #get i-th importance score
            vals_box[:,i] = self.get_score(preds, l1)
        
        #optionally log update
        if self.verbose is True:
            print("[MVS-SA]: Sample importance calculated")
            
        #return all scores for each tree to see the change of importance during training
        if return_trajectory is True:
            return vals_box
        #return final importance score as the sum across all iterations
        else:
            return np.sum(vals_box, axis=1)
    
        
