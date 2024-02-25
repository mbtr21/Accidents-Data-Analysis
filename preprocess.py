from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self, dataframe):
        self.data_frame = dataframe

    def set_index(self, index):
        self.data_frame.set_index(index, inplace=True)

    def na_handler(self, args=None):
        if args is not None:
            self.data_frame = self.data_frame.fillna(method=args)
        else:
            self.data_frame = self.data_frame.dropna(axis=0, how='any')

    def standardization(self, columns=None):
        if columns is not None:
            data = self.data_frame[columns].copy()
            data = data.loc[:, columns]
            scaled_data = (data - data.mean(axis=0)) / data.std(axis=0)
            self.data_frame[columns] = scaled_data

    def one_hot_encoder(self, columns):
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(self.data_frame[[columns]])
        one_hot_encoded = encoder.transform(self.data_frame[[columns]])
        encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out())
        encoded_df.index = self.data_frame.index
        self.data_frame = pd.concat([self.data_frame.drop(columns, axis=1), encoded_df], axis=1)


class FeatureSelector:
    def __init__(self, dataframe):
        self.data_frame = dataframe

    def reduction_dimension_by_pca(self, scaled_columns, handel='return', index=''):
        pca = PCA()
        pca_features = pca.fit_transform(self.data_frame.loc[:, scaled_columns])
        component_names = [f"PC{i + 1}" for i in range(pca_features.shape[1])]
        pca_features = pd.DataFrame(pca_features, columns=component_names)
        print(pca_features)
        if handel == 'merge':
            self.data_frame.reset_index(inplace=True, drop=True)
            self.data_frame = pd.concat([self.data_frame, pca_features], axis=1)
            self.data_frame.set_index(index, inplace=True)
        elif handel == 'replace':
            self.data_frame = pd.concat([self.data_frame['class'], pca_features], axis=1)
        else:
            return pca_features

    def calculate_mutual_inf(self, target, number_of_features):
        features = self.data_frame.copy()
        object_columns = self.data_frame.select_dtypes(include='object')
        features.drop(columns=object_columns, inplace=True)
        discrete_features = features.dtypes == int
        mi_scores = mutual_info_classif(features, target, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=features.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores[:number_of_features]

    def extend_data_by_k_means(self, features, numbers_of_cluster):
        data = self.data_frame.copy()
        data_selected = data.loc[:, features]
        k_means_scores = list()
        for number in numbers_of_cluster:
            k_means = KMeans(number, random_state=42)
            y = k_means.fit_predict(data_selected)
            score = silhouette_score(data_selected, y)
            k_means_scores.append([k_means, score, y])
        k_means_scores = sorted(k_means_scores, key=lambda x: x[1], reverse=True)
        print(k_means_scores)
        k_means = k_means_scores.pop()
        self.data_frame["cluster"] = k_means[2]


