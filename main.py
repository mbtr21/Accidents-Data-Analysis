import pandas as pd
from hypothesis_tests import HypothesisTests
from preprocess import DataPreprocessor, FeatureSelector
from predictor import SvmClassifier
from dashboard import AccidentRoadDashboard


def initial():
    df = pd.read_csv('data.csv')
    data_preprocessor = DataPreprocessor(df)
    data_preprocessor.na_handler()
    data_preprocessor.drop_columns(columns=['accident_year', 'lsoa_of_casualty', 'status'
                                            , 'accident_reference', 'age_band_of_casualty'])
    data_preprocessor.set_index('accident_index')
    hypothesis_test = HypothesisTests()
    female = data_preprocessor.data_frame.loc[data_preprocessor.data_frame['sex_of_casualty'] == 2]['casualty_severity']
    male = data_preprocessor.data_frame.loc[data_preprocessor.data_frame['sex_of_casualty'] == 1]['casualty_severity']
    gender_t_test = hypothesis_test.two_t_test(female, male)
    gender_t_test_pvalue = gender_t_test.pvalue
    driver = data_preprocessor.data_frame.loc[data_preprocessor.data_frame['casualty_class'] == 1]['casualty_severity']
    passenger = data_preprocessor.data_frame.loc[data_preprocessor.data_frame['casualty_class'] == 2]['casualty_severity']
    pedestrian = data_preprocessor.data_frame.loc[data_preprocessor.data_frame['casualty_class'] == 3]['casualty_severity']
    anova_test = hypothesis_test.anova_test(data_groups=[driver, passenger, pedestrian])
    anova_test_f_value = anova_test.pvalue
    # data_preprocessor.data_frame.to_csv('new_data.csv')
    target = data_preprocessor.data_frame.pop('casualty_severity')
    data_preprocessor.standardization(columns=data_preprocessor.data_frame.columns)
    feature_engineering = FeatureSelector(data_preprocessor.data_frame)
    feature_engineering.extend_data_by_k_means(features=feature_engineering.data_frame.columns
                                               , numbers_of_cluster=[3])
    feature_engineering.reduction_dimension_by_pca(scaled_columns=feature_engineering.data_frame.columns
                                                   , handel='return', index='accident_index')
    shape = feature_engineering.data_frame.shape
    features = feature_engineering.calculate_mutual_inf(target=target
                                                        , number_of_features=int(0.8 * shape[1]))
    features = feature_engineering.data_frame.loc[:, list(features.index)]
    svm = SvmClassifier()
    svm.set_train_test(features, target, test_size=0.3, random_state=42)
    svm.train_model(kernel='linear', C=0.3)
    f1_score = svm.result_model()[0]
    return f1_score,anova_test_f_value, gender_t_test_pvalue


if __name__ == '__main__':
    f1_score,t_test,anova_test = initial()
    df = pd.read_csv('new_data.csv')
    df['sex_of_casualty'] = df['sex_of_casualty'].replace(1, 'male')
    df['sex_of_casualty'] = df['sex_of_casualty'].replace(2, 'female')
    df = df[(df['sex_of_casualty'] == 'male') |
            (df['sex_of_casualty'] == 'female')]
    df['casualty_class'] = df['casualty_class'].replace(1, 'driver')
    df['casualty_class'] = df['casualty_class'].replace(2, 'passenger')
    df['casualty_class'] = df['casualty_class'].replace(3, 'pedestrian')
    dash = AccidentRoadDashboard(df=df, t_test=t_test, anova_test=anova_test, kpi=f1_score)
    dash.app.run()



