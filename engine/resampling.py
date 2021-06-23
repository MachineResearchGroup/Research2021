from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek


class Resampling:

    def __init__(self, name):
        self.strategie = None
        self.name = name

        if name == "enn":
            self.strategie = EditedNearestNeighbours(sampling_strategy='auto',
                                                     n_neighbors=3,
                                                     kind_sel='all',
                                                     n_jobs=-1)
        elif name == "allknn":
            self.strategie = AllKNN(sampling_strategy='auto',
                                    n_neighbors=3,
                                    kind_sel='all',
                                    allow_minority=False,
                                    n_jobs=-1)
        elif name == "renn":
            self.strategie = RepeatedEditedNearestNeighbours(sampling_strategy='auto',
                                                             n_neighbors=3,
                                                             max_iter=100,
                                                             kind_sel='all',
                                                             n_jobs=-1)

        elif name == "tomek":
            self.strategie = TomekLinks(sampling_strategy='auto',
                                        n_jobs=-1)

        elif name == "smote":
            self.strategie = SMOTE(sampling_strategy='auto',
                                   k_neighbors=5,
                                   n_jobs=-1)

        elif name == "bdsmote":
            self.strategie = BorderlineSMOTE(n_jobs=-1)

        elif name == "adasyn":
            self.strategie = ADASYN(sampling_strategy='auto',
                                    n_neighbors=5,
                                    n_jobs=-1)

        elif name == "smoteenn":
            self.strategie = SMOTEENN(sampling_strategy='auto',
                                      smote=None,
                                      enn=None,
                                      n_jobs=-1)

        elif name == "smotetomek":
            self.strategie = SMOTETomek(sampling_strategy='auto',
                                        smote=None,
                                        tomek=None,
                                        n_jobs=-1)

    def fit_resample(self, x, y):
        x_res, y_res = self.strategie.fit_resample(x, y)
        return x_res, y_res
