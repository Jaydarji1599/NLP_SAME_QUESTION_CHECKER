import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data

import warnings
warnings.filterwarnings('ignore')



if __name__ == '__main__':
    dataset = pd.read_csv('data/train.csv')
    dataset = dataset.dropna()
    dataset['question1'] = dataset['question1'].apply(data.preprocess)
    dataset['question2'] = dataset['question2'].apply(data.preprocess)
    dataset_df = data.data_engineering(dataset)
    print(dataset_df.head())