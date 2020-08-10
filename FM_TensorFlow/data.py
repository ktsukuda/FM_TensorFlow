import csv
import numpy as np


class DataSplitter():

    def __init__(self):
        self.feature_idx = dict()
        self.train, self.validation, self.test = [], [], []
        self._load_data()

    def _load_data(self):
        i = 0
        for data_type, data_dict in {"train": self.train, "validation": self.validation, "test": self.test}.items():
            print('Load {} data...'.format(data_type))
            with open('data/ml-tag/ml-tag.{}.libfm'.format(data_type)) as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    features = []
                    for feature in row[1:]:
                        if feature not in self.feature_idx:
                            self.feature_idx[feature] = i
                            i += 1
                        features.append(self.feature_idx[feature])
                    data_dict.append([[float(row[0])], features])
        self.train = np.array(self.train)
        self.validation = np.array(self.validation)
        self.test = np.array(self.test)

    @property
    def n_feature(self):
        return len(self.feature_idx)
