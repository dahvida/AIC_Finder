"""Fluorescence overlap script.

Pipeline to evaluate how many false positives found by MVS-A are also predicted to be
autofluorescent by a multitask neural network.

Before running the script, make sure that you have already ran eval_pipeline.py with
logging enabled, since those .csv files will be used for this analysis. Also, make sure
to unzip "cleaned_autofluo.zip" in ../Misc. 
If no flags are passed, the analysis will use the default parameters used in our study. 
The output will be saved as ../Results/fluo_overlap.csv

Steps:
    1. Load preprocessed fluorescence dataset from ../Misc
    2. Calculate mordred descriptors for training set
    3. Preprocess mordred descriptors
    4. Train multitask neural network
    5. For each logged dataset:
        5.1. Load and featurize FPs predicted by MVS-A
        5.2. Preprocess features
        5.3. Calculate predictions with neural network
        5.4. Calculate overlap by dividing number of predicted fluorescent versus
             total MVS-A FPs
        5.5. Compute false positive detection metrics
    6. Save
"""

import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras import backend as K
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from utils import *
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import os
import argparse

###############################################################################

parser = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--units', default=512, type=int,
                    help="Dense layers units")

parser.add_argument('--dropout', default=0.15, type=float,
                    help="Dropout temperature")

parser.add_argument('--activation', default="swish",
                    help="Dense layer activation function")

parser.add_argument('--tau', default=2.0, type=float,
                    help="Logit-adjusted loss class imbalance coefficient")

parser.add_argument('--lr', default=0.01, type=float,
                    help="Initial learning rate")

parser.add_argument('--decay_rate', default=0.95, type=float,
                    help="Decay coefficient for the learning rate")

parser.add_argument('--decay_period', default=5, type=int,
                    help="Epoch interval between learning rate reductions")

parser.add_argument('--batch_size', default=64, type=int,
                    help="Batch size")

parser.add_argument('--epochs', default=100, type=int,
                    help="Number of epochs")

parser.add_argument('--threshold', default=0.5, type=int,
                    help="Value to use when converting predicted probabilities in labels")

args = parser.parse_args()

###############################################################################

def get_mordred(mols):
    """
    Helper function to calculate 2D Mordred descriptors from
    RDKIT molecules
    """
    calc = Calculator(descriptors, ignore_3D=True)
    output = calc.map(mols, quiet=True)
    output = [x for x in output]
    
    descs = np.zeros((len(mols), 1613))
    for i in range(len(mols)):
        descs[i,:] = np.array(output[i])
    
    descs = np.nan_to_num(descs, neginf=-10000, posinf=10000)
    descs[descs < -10000] = -10000
    descs[descs > 10000] = 10000

    return descs

# -----------------------------------------------------------------------------#

class LnBnDr(tf.keras.Model):
    """
    Custom dense layer using the following structure:
    Linear -> BatchNorm -> Dropout -> Activation
    Taken from fastai tutorials on tabular data
    """
    def __init__(self,
                 units,
                 dropout = 0.2,
                 activation = "relu",
                 ):
        super().__init__(self)
        self.dropout = layers.Dropout(dropout)
        self.batchN = layers.BatchNormalization()
        self.dense = layers.Dense(units, activation=None,
                                 )
        self.activation = layers.Activation(activation)
        
    def call(self,
             x, 
             training=True):   
        
        x = self.dense(x)
        x = self.batchN(x, training=training)
        x = self.dropout(x, training=training)
        return self.activation(x)

# -----------------------------------------------------------------------------#
    
class multitask_network(tf.keras.Model):
    """
    Multitask network architecture using LnBnDr layers, Logit-adjusted
    loss and class weighting to tackle class imbalance.
    """
    def __init__(self,
                 u,
                 drop,
                 act,
                 tau,
                 num_classes
                 ):
        super().__init__(self)
        
        self.num_classes = num_classes
        
        self.first = LnBnDr(u, drop, act)
        self.second = LnBnDr(u, drop, act)
        self.final = layers.Dense(num_classes, activation=None)
        self.act = layers.Activation("sigmoid")
        
        self.class_weights = np.full((num_classes,), fill_value=1)    
        self.pi= np.full((num_classes,), fill_value=1)  
        self.tau = tau
        
    def call(self, x, training=True):   
        
        x = self.first(x, training=training)
        x = self.second(x, training=training)
        x = self.final(x)
        return self.act(x)
    
    def set_class_weights(self, y):
        
        for i in range(y.shape[1]):
            ratio = y.shape[0] / np.sum(y[:,i])
            self.class_weights[i] = np.sqrt(ratio) - 1
        self.pi = 1 / self.class_weights
            
    def train_step(self, batch):
        x = batch[0]
        y = batch[1]
        
        with tf.GradientTape() as tape:
            
            x = self.first(x, True)
            x = self.second(x, True)
            x = self.final(x)
            
            shift_1 = tf.math.multiply(y, self.pi)
            shift_2 = tf.math.multiply(tf.math.subtract(1.0, y), (1.0 - self.pi))
            shift_1 = tf.cast(shift_1, dtype=tf.float32)
            shift_2 = tf.cast(shift_2, dtype=tf.float32)
            
            shift = shift_1 + tf.math.abs(shift_2)
            shift = tf.math.log(shift ** self.tau)
            probs = self.act(x + shift)
            
            probs = tf.expand_dims(probs, -1)
            y = tf.expand_dims(y, -1)
            batch_loss = tf.keras.metrics.binary_crossentropy(y, probs)
            aggregated = 0
           
            for i in range(self.num_classes):
                weight_i = (K.cast(K.equal(y[:,i], 1), K.floatx()) * self.class_weights[i]) + 1
                scaled = tf.multiply(batch_loss[:,i], weight_i) 
                aggregated += tf.reduce_mean(scaled) / self.num_classes
                
        gradients = tape.gradient(aggregated, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {
            "loss": aggregated
            }

##############################################################################

def main(
        units,
        dropout,
        activation,
        tau,
        lr,
        decay_rate,
        decay_period,
        batch_size,
        epochs,
        threshold
        ):
    
    print("[fluo]: Starting analysis...")
    
    #load csv and get rdkit mols
    db = pd.read_csv("../Misc/cleaned_autofluo.csv", index_col=0)
    mols = [Chem.MolFromSmiles(x) for x in list(db["SMILES"])]
    
    #calculate all 2D Mordred descs and get Y matrix
    descs = get_mordred(mols)
    db.drop("SMILES", axis=1, inplace=True)
    y = np.array(db)
    print("[fluo]: Training set generated")
    
    #set up learning rate decay scheduler
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=np.floor(8096 / batch_size) * decay_period,
            decay_rate=decay_rate,
            staircase=True)
    
    #prune descriptors with low variance and normalize
    variance = VarianceThreshold(0.05)
    variance.fit(descs)
    descs = variance.transform(descs)
    scaler = StandardScaler()
    descs = scaler.fit_transform(descs)
    
    #create multitask neural network
    model = multitask_network(units,
                              dropout,
                              activation,
                              tau,
                              6)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer)

    #set class weights
    model.set_class_weights(y)

    #train network
    model.fit(descs, y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0)
    print("[fluo]: Neural network trained")
    
    #get all logged dataset names
    names = os.listdir("../Logs/eval/")
    output = np.zeros((len(names), 4))
    
    #loop over all logged datasets
    for i in range(len(names)):
        
        print(f"Processing dataset: {names[i]}")

	#load csv and select all compounds flagged by MVS-A as FP 
        temp = pd.read_csv("../Logs/eval/" + names[i])
        y = np.array(temp["False positives"])
        idx = np.array(temp["FP - mvsa"])
        idx = np.where(idx == 1)[0]
        smiles = list(temp["SMILES"])
        mols = [Chem.MolFromSmiles(x) for x in smiles]

        #compute and preprocess descriptors for new csv
        val = get_mordred(mols)
        val = variance.transform(val)
        val = scaler.transform(val)

        #get predictions
        preds = model.predict(val)
        preds = np.max(preds, axis=1)
        
        #convert probs in labels using threshold
        p1 = preds[idx].copy()
        p1[p1 > threshold] = 1
        p1[p1 <= threshold] = 0
        
        #compute overlap
        output[i, 0] = np.sum(p1) * 100 / p1.shape[0]
        
        #compute metrics
        p2 = preds.copy()
        t2 = np.percentile(preds, 90)
        p2[p2 > t2] = 1
        p2[p2 <= t2] = 0
        output[i,1] = precision_score(y, p2)
        output[i,2] = enrichment_factor_score(y, p2)
        output[i,3] = bedroc_score(y, preds, reverse=True)
        
        print("--------------------------")
    
    #store in pandas dataframe and save
    output = pd.DataFrame(
            data = output,
            index = names,
            columns = ["Overlap %", "Precision@10", "EF@10", "BEDROC"]
            )
    output.to_csv("../Results/fluo_output.csv")
    print("[fluo]: Analysis finished, file saved at ../Results/fluo_output.csv")


if __name__ == "__main__":
    main(
            args.units,
            args.dropout,
            args.activation,
            args.tau,
            args.lr,
            args.decay_rate,
            args.decay_period,
            args.batch_size,
            args.epochs,
            args.threshold
            )



