import pandas as pd
from engine.dataset import DataSet

if __name__ == '__main__':
    dataset = DataSet()
    for i in range(10):
        dataset.define_datasets(index_data=i+1, data=pd.read_csv('../dataset/PROMISE_exp_preprocessed.csv', encoding='utf-8'))

