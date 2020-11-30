import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

class Preprocessing:
    
    @staticmethod
    def data_info(df):
        info = {}

        info['shape'] = df.shape
        info['describe'] = df.describe()
        info['info'] = df.info()
        info['types'] = df.dtypes

        return info

    @staticmethod
    def select_features(df, target):
        x = df
        y = df[target]
        
        selector = SelectKBest(f_regression, k=8)
        selector.fit_transform(x, y)

        features = x.columns[selector.get_support(indices=True)].to_list()

        df_train = df[features]
        df_train[target] = df[target]

        return df_train
        


    
    
