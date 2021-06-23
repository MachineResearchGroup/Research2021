import csv

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
from engine import classifiers
from tools import data_tool


def box(data_name):
    x = 'Algoritmo'
    y = 'F1'
    hue = 'Algorithm'
    data = data_tool.get_general_cv_mean(data_name)
    sns.set_theme(style='whitegrid')
    sns.boxplot(x=x, y=y, data=data, saturation=100, width=0.9)
    plt.show()


def matrix(classifier, X_test, y_test, data_name, n_inter):
    labels = ['A', 'FT', 'L', 'LF', 'MN', 'O', 'PE', 'PO', 'SC', 'SE', 'US']
    nameClf = classifiers.getClf_Name(classifier.__class__)
    plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues, display_labels=labels)
    plt.title(nameClf+'+'+data_name+' Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    path = '../results/images/matrix/'+data_name+'/confusionMatrix-'+str(n_inter)+'('+nameClf+')'
    plt.savefig(path, format='png')


def line(data_name):
    x = 'Algoritmo'
    y = 'F1'
    hue = 'Algorithm'
    data = data_tool.get_general_cv_mean(data_name)
    sns.set_theme(style='whitegrid')
    sns.lineplot(x=x, y=y, data=data, palette="tab10", linewidth=2.5)
    plt.show()

def bar(data_name):
    x = 'Algorithm'
    y = 'F1'
    hue = 'Algorithm'
    data = data_tool.get_general_cv_mean(data_name)
    sns.set_theme(style='whitegrid')
    ax = sns.barplot(x=x, y=y, data=data, saturation=100, width=0.9)
    ax.set_title('Barplot ' + data_name)
    plt.show()

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


names = {'origin': 'Original', 'tomek': 'Tomek Links', 'smote': 'SMOTE', 'bdsmote': 'SMOTE-Borderline_1', 'adasyn': 'ADASYN', 'smotetomek': 'SMOTE-TL'}


if __name__ == '__main__':
    for n in names:
        box(n)