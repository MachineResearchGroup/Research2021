import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import tools.data_tool as bt
from sklearn.model_selection import StratifiedShuffleSplit

from engine.resampling import Resampling


class DataSet:

    def define_datasets(self, index_data, data):
        resamplings = ['origin', 'tomek', 'smote', 'bdsmote', 'adasyn', 'smotetomek']
        k_folds = 6
        X = data['RequirementText']
        y = data['Class']
        cv = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.2)
        indices = cv.split(X, y)

        i = 1
        for train, test in indices:
            vetorClas = LabelEncoder().fit(y[train])
            y_train = vetorClas.transform(y[train])
            y_test = vetorClas.transform(y[test])

            vetorText = TfidfVectorizer().fit(X[train])
            x_train = vetorText.transform(X[train])
            x_test = vetorText.transform(X[test])

            self.export_data_test(index_data, i, x_test, y_test)

            for resampling in resamplings:
                self.export_data_train(index_data, resampling, i, x_train, y_train)
            i += 1
    def export_data_train(self, index_data, resampling, index, x_train, y_train):
        path = '../results/datasets/data_'+str(index_data)+'/train/'+resampling + '_train(' + str(index) + ')'
        if resampling != 'origin':
            x_train, y_train = Resampling(resampling).fit_resample(x_train, y_train)

        bt.save_sparse_csr(path + '.npz', x_train)

        df = {'Class': y_train}
        df = pd.DataFrame(df)
        df.to_csv(path + '.csv')

        self.update_dataset_detail(index_data, resampling, y_train, index)

    def export_data_test(self, index_data, index, x_test, y_test):
        path = '../results/datasets/data_'+str(index_data)+'/test/test(' + str(index) + ')'

        bt.save_sparse_csr(path+'.npz', x_test)

        df = {'Class': y_test}
        df = pd.DataFrame(df)
        df.to_csv(path+'.csv')

        self.update_dataset_detail(index_data, None, y_test, index, test=True)

    def get_data(self, index_data, resampling, num):
        train_tfidf = bt.load_sparse_csr('../results/datasets/data_'+str(index_data)+'/train/'+resampling + '_train(' + str(num) + ').npz')
        train_class = pd.read_csv('../results/datasets/data_'+str(index_data)+'/train/'+resampling + '_train(' + str(num) + ').csv')
        test_tfidf = bt.load_sparse_csr('../results/datasets/data_'+str(index_data)+'/test/test(' + str(num) + ').npz')
        test_class = pd.read_csv('../results/datasets/data_'+str(index_data)+'/test/test(' + str(num) + ').csv')

        return train_tfidf, train_class, test_tfidf, test_class

    def update_dataset_detail(self, index_data, resampling, y, iter, test=False):
        path = '../results/datasets/data_'+str(index_data)+'/detail/dataset_detail(' + str(iter) + ').csv'

        file = Path(path)

        classes = ['A', 'FT', 'L', 'LF', 'MN', 'O', 'PE', 'PO', 'SC', 'SE', 'US', 'total']
        counts = self.count_classes(y)

        if file.is_file():
            df = pd.read_csv(path)
            try:
                os.remove(file)
            except OSError as e:
                print(e)

            df[resampling] = counts
            df.to_csv(path, index=False)

        else:
            if test:
                df = pd.DataFrame({'classes': classes,
                                   'test': counts})
            else:
                df = pd.DataFrame({'classes': classes,
                                   resampling: counts})
            df.to_csv(path, index=False)

    def count_classes(self, y):
        y = pd.DataFrame(y)
        values = y.value_counts().to_dict()
        count = []

        for i in range(11):
            count.append(values[(i,)])
        total = sum(count)
        count.append(total)

        return count

