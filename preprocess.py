from sklearn.metrics import silhouette_score  # Import silhouette_score for cluster analysis
from sklearn.preprocessing import OneHotEncoder  # Import OneHotEncoder for categorical variable encoding
from sklearn.feature_selection import mutual_info_classif  # Import mutual_info_classif for feature selection
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from sklearn.cluster import KMeans  # Import KMeans for clustering
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations


class DataPreprocessor:
    def __init__(self, dataframe):
        self.data_frame = dataframe  # Initialize with a dataframe

    def set_index(self, index):
        self.data_frame.set_index(index, inplace=True)  # Set the dataframe index

    def drop_columns(self, columns):
        self.data_frame.drop(columns=columns, inplace=True)  # Drop specified columns

    def na_handler(self, args=None):
        # Handle missing values either by filling or dropping
        if args is not None:
            self.data_frame = self.data_frame.fillna(method=args)
        else:
            self.data_frame = self.data_frame.dropna(axis=0, how='any')

    def standardization(self, columns=None):
        # Standardize data (mean=0, std=1) for specified columns
        if columns is not None:
            data = self.data_frame[columns].copy()
            data = data.loc[:, columns]
            scaled_data = (data - data.mean(axis=0)) / data.std(axis=0)
            self.data_frame[columns] = scaled_data

    def one_hot_encoder(self, columns):
        # Apply one-hot encoding to specified columns
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(self.data_frame[[columns]])
        one_hot_encoded = encoder.transform(self.data_frame[[columns]])
        encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out())
        encoded_df.index = self.data_frame.index
        self.data_frame = pd.concat([self.data_frame.drop(columns, axis=1), encoded_df], axis=1)

class FeatureSelector:
    def __init__(self, dataframe):
        self.data_frame = dataframe  # Initialize with a dataframe

    def reduction_dimension_by_pca(self, scaled_columns, handel='return', index=''):
        # Reduce dimensionality using PCA
        pca = PCA()
        pca_features = pca.fit_transform(self.data_frame.loc[:, scaled_columns])
        component_names = [f"PC{i + 1}" for i in range(pca_features.shape[1])]
        pca_features = pd.DataFrame(pca_features, columns=component_names)
        if handel == 'merge':
            # Merge PCA features with original dataframe
            self.data_frame.reset_index(inplace=True, drop=False)
            self.data_frame = pd.concat([self.data_frame, pca_features], axis=1)
            self.data_frame.set_index(index, inplace=True)
        elif handel == 'replace':
            # Replace original features with PCA features
            self.data_frame = pd.concat([self.data_frame['class'], pca_features], axis=1)
        else:
            # Return PCA features only
            self.data_frame = pca_features

    def calculate_mutual_inf(self, target, number_of_features):
        # Calculate and return mutual information scores for features
        features = self.data_frame.copy()
        features.reset_index(inplace=True, drop=True)
        object_columns = features.select_dtypes(include='object')
        features.drop(columns=object_columns, inplace=True)
        discrete_features = features.select_dtypes(include='number')
        mi_scores = mutual_info_classif(y=target, X=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=features.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores[:number_of_features]

    def extend_data_by_k_means(self, features, numbers_of_cluster):
        # Apply KMeans clustering and extend dataframe with cluster information
        data = self.data_frame.copy()
        data.reset_index(inplace=True, drop=True)
        data_selected = data.loc[:, features].astype(np.float32)
        k_means_scores = list()
        for number in numbers_of_cluster:
            k_means = KMeans(number, random_state=42)
            y = k_means.fit_predict(data_selected)
            score = silhouette_score(data_selected, y)
            k_means_scores.append([k_means, score, y])
        k_means_scores = sorted(k_means_scores, key=lambda x: x[1], reverse=True)
        k_means = k_means_scores.pop()
        self.data_frame["cluster"] = k_means[2]

