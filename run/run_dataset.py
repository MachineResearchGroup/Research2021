import pandas as pd
from models.dataset import DataSet

if __name__ == '__main__':
    dataset = DataSet()

    dataset.define_datasets(qtd=30, data=pd.read_csv('../dataset/PROMISE_exp_preprocessed.csv', encoding='utf-8'))

