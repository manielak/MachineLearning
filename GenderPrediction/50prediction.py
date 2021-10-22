import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

import qgrid

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_hdf('../input/train.adult.h5')
