import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import ClassifierChain
from numpy import *
import random


num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest(BaseModel):
    def __init__(self) -> None:
        super(RandomForest, self).__init__()

    def build_chain_model():
        base_model = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        chain_model = ClassifierChain(base_model, order='random', random_state=42)
        return base_model, chain_model
    
    def data_transform(self) -> None:
        ...
