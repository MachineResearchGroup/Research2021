from engine.evaluation import Evaluate
from tools import data_tool


def run_(resamplings):
    for resampling in resamplings:
        eval = Evaluate(7, 6, resampling)
        eval.run()


if __name__ == '__main__':
    resamplings = ['origin', 'tomek', 'adasyn', 'smote', 'bdsmote', 'smotetomek']
    run_(resamplings)


