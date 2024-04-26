import numpy as np
import pandas as pd
from Config import *
import random
from sklearn.feature_extraction.text import TfidfVectorizer

seed =0
random.seed(seed)
np.random.seed(seed)


def get_embeddings(tr:pd.DataFrame, ts:pd.DataFrame):
   return __tfidf_embd(tr, ts)

def __tfidf_embd(tr, ts):
    vectorizer = TfidfVectorizer(stop_words='english')
    train = vectorizer.fit_transform(tr)
    test = vectorizer.transform(ts)
    return train, test