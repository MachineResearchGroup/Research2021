from collections import OrderedDict
import collections

import joblib
import pandas as pd
from sklearn.svm import SVC as SVM
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import PassiveAggressiveClassifier as PA

clf_Name = {ET: 'ET', LR: 'LR', MLP: 'MLP', MNB: 'MNB', PA: 'PA', SGD: 'SGD', SVM: 'SVM'}
#clf_Prt = {ET: [], LR: [], MLP: [], MNB: [], PA: [], SGD: [], SVM: []}

def getClf(interaction, resampling):
    clf_Prt = {ET: [], LR: [], MLP: [], MNB: [], PA: [], SGD: [], SVM: []}
    for clf in clf_Name:
        params = get_params(interaction, resampling, clf_Name[clf])
        instanciamento(clf_Prt, clf, params)
    return clf_Prt


def get_params(interaction, resampling, algorithm):
    params = pd.read_csv('../results/hyperparametrization/data_'+str(interaction)+'/'+resampling+'/hypeResultsBayesSearchCV(' + algorithm + ').csv')
    return params['Params']

# def getClf(interaction, fold, resampling):
#     for clf in clf_Prt:
#         classifier = get_model(interaction, fold, resampling, clf)
#         clf_Prt[clf].append(classifier)
#     return clf_Prt
#
#
# def get_params(index_data, resampling, algorithm):
#     params = pd.read_csv('../results/hyperparametrization/data_'+str(index_data)+'/'+resampling+'/hypeResultsBayesSearchCV(' + algorithm + ').csv')
#     return params['Params']


def instanciamento(clf_Prt, _class, params):
    for param in params:
        param = dict(eval(param))
        _classifier = _class(**param)
        clf_Prt[_class].append(_classifier)


def getClf_Name(classifier):
    return clf_Name[classifier]


def get_model(interaction, fold, resampling, clf):
    return joblib.load('../results/hyperparametrization/models/data_'+str(interaction)+'/'+resampling+'/'+clf.__name__+'('+str(fold)+').joblib.pkl')

