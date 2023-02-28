import lightgbm as lgb
import numpy as np
from sklearn.metrics import *

class sample_analysis:
    """Implementation of MVS-A for sample importance analysis.
    
    This class provides all methods necessary to use MVS-A to
    determine which samples are most influential during the training
    of LightGBM classifiers. These importance estimates can be
    used to identify false positives and true positives in the 
    training set without the need of a clean validation set.
    
    Attributes:
        x:              array of molecular features (M, K) for the
                        training set, i.e. ECFPs
        y:              array of training set labels (M,)
        params:         hyperparameter dict for the LightGBM classifier
        verbose:        whether to print info during each step
        seed:           random seed for reproducibility

        model:          trained LightGBM model (set by self.get_model)
        n_leaves:       number of leaves for each tree in the ensemble
                        (set by self.get_model)
        n_estimators:   number of trees in the ensemble (set by
                        self.get_model)
    """

    def __init__(self,
            x: np.ndarray,
            y: np.ndarray,
            params: dict = None,
            verbose: bool = True,
            seed: int = 0
            ) -> None:
        """Instantiates class object according to training dataset and
        user preferences

        Args:
            x:              array of molecular features (M, K) for the
                            training set, i.e. ECFPs
            y:              array of training set labels (M,)
            params:         hyperparameter dict for the LightGBM classifier
            verbose:        whether to print info during each step
            seed:           random seed for reproducibility

        Returns:
            None
        """

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
                
        
    def get_model(self) -> None:
        """Trains LightGBM classifier on loaded training set and
        saves it in the class
        
        Args:
            None

        Returns:
            None
        """

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
    
    def get_l1(self,
            tree_n: int
            ) -> float:
        """Computes L1 norm of leaf weights for i-th tree, which
        will normalize square hessians when computing MVS scores
        
        Args:
            tree_n: tree ID inside the ensemble

        Returns:
            L1 norm of leaf weights for i-th tree
        """

        #get n_leaves for n-th tree
        n = self.n_leaves[tree_n]
        l1 = 0

        #calculate L1 norm for n-th tree
        for i in range(n):
            l1 += np.abs(self.model.get_leaf_output(tree_n, i))

        return l1
        
    def get_score(self,
            preds: np.ndarray,
            l1: float
            ) -> np.ndarray:
        """Computes MVS scores for i-th iteration
        
        Args:
            preds:  array (M,) of predictions at the i-th
                    iteration (sum of single tree preds up
                    to i)
            l1:     L1 norm of leaf weights for the last tree

        Returns:
            array (M,) of MVS scores for all samples at the
            i-th iteration
        """

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
    
    def get_importance(self,
            return_trajectory: bool = False
            ) -> np.ndarray:
        """Computes global MVS-A score for all samples
        
        Args:
            return_trajectory:  defines whether to keep all MVS scores
                                for each iteration or to sum them up
        
        Returns:
            An array (M,) defining the global importance across
            the training process for all samples. If return_trajectory
            is True, it returns an array (M,T), where T = self.n_estimators.
            This can be useful if you want to monitor the importance
            change of given samples across training.
                                
        """
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
    
        
