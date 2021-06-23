import csv
import warnings
import numpy as np
import pandas as pd
# from tools import plot
from engine import classifiers
from engine.dataset import DataSet
from sklearn.metrics import precision_score, recall_score, f1_score


class Evaluate:

    def __init__(self, interaction, k_fold, resampling):
        self.resampling = resampling
        self.k_fold = k_fold
        self.interaction = interaction

    def run(self):
        warnings.filterwarnings('ignore')
        final_mean = get_structure_results()
        print('-- '+self.resampling+' --')
        for index in range(self.interaction):
            cv_mean = get_structure_results()
            clfs = classifiers.getClf(index + 1, self.resampling)
            print(repr(index + 1) + ' interaction')
            for i in range(self.k_fold):
                train_tfidf, train_class, test_tfidf, test_class = DataSet().get_data(index+1, self.resampling, i+1)
                print(repr(i + 1) + ' dobra')

                for model in clfs:
                    print(classifiers.getClf_Name(model))
                    model = clfs[model][i]
                    model.fit(train_tfidf, np.array(train_class['Class']))
                    # plot.matrix(model, test_tfidf, np.array(test_class['Class']), self.resampling, i+1)
                    y_pred = model.predict(test_tfidf)

                    precision, recall, f1 = get_results(np.array(test_class['Class']), y_pred)

                    score = {'Algoritmo': classifiers.getClf_Name(model.__class__), 'interaction': str(i+1), 'Precision': precision,
                             'Recall': recall, 'F1': f1}

                    cv_mean[classifiers.getClf_Name(model.__class__)]['Precision'] += precision
                    cv_mean[classifiers.getClf_Name(model.__class__)]['Recall'] += recall
                    cv_mean[classifiers.getClf_Name(model.__class__)]['F1'] += f1
                    path = '../results/evaluation/data_' + str(index + 1) + '/score(' + self.resampling + ').csv'
                    gravacaoCSV(path, score)

            cv_mean = calcularMedia(self.k_fold, cv_mean)
            for clf in cv_mean:
                precision = cv_mean[clf]['Precision']
                recall = cv_mean[clf]['Recall']
                f1 = cv_mean[clf]['F1']

                final_mean[clf]['Precision'] += precision
                final_mean[clf]['Recall'] += recall
                final_mean[clf]['F1'] += f1

                helper_cv_mean = {'Algoritmo': clf, 'Precision': precision, 'Recall': recall, 'F1': f1}
                helper_general_cv_mean = {'Algoritmo': clf, 'Interaction': index+1, 'Precision': precision, 'Recall': recall, 'F1': f1}
                path = '../results/evaluation/data_' + str(index + 1) + '/cv_mean(' + self.resampling + ').csv'
                gravacaoCSV(path, helper_cv_mean)
                path = '../results/evaluation/general_cv_mean(' + self.resampling + ').csv'
                gravacaoCSV(path, helper_general_cv_mean)
        final_mean = calcularMedia(self.interaction, final_mean)
        path = '../results/evaluation/final_mean(' + self.resampling +').csv'
        for clf in final_mean:
            precision = final_mean[clf]['Precision']
            recall = final_mean[clf]['Recall']
            f1 = final_mean[clf]['F1']

            helper_final_mean = {'Algoritmo': clf, 'Precision': precision, 'Recall': recall, 'F1': f1}
            gravacaoCSV(path, helper_final_mean)



def gravacaoCSV(path, dicio):
    try:
        open(path, 'r')
        with open(path, 'a') as arq:
            writer = csv.writer(arq)
            writer.writerow(dicio.values())
    except IOError:
        dataF = pd.DataFrame(columns=dicio.keys())
        dataF = dataF.append(dicio, ignore_index=True)
        dataF.to_csv(path, index=False)


def calcularMedia(divisor, dicio):
    for algoritmo in dicio:
        for metrica in dicio[algoritmo]:
            dicio[algoritmo][metrica] = dicio[algoritmo][metrica] / divisor
    return dicio


def get_structure_results():
     return {'ET': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0, 'G-mean': 0.0},
            'LR': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0, 'G-mean': 0.0},
            'MLP': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0, 'G-mean': 0.0},
            'MNB': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0, 'G-mean': 0.0},
            'PA': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0, 'G-mean': 0.0},
            'SGD': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0, 'G-mean': 0.0},
            'SVM': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}}


def get_results(y_pred, y_test):
    precisao = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    return precisao, recall, f1


def remove(dicio, key_remove):
    new_dict = {}
    for key, value in dicio.items():
        if key is not key_remove:
            new_dict[key] = value

    return new_dict

