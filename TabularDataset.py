import pandas as pd
import numpy as np
import folktables
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from folktables import ACSDataSource
from datasets import acs_income, acs_employment, compas

dataset_params = {
    'ACSIncome': acs_income,
    'ACSEmployment': acs_employment,
    'COMPAS': compas
}

class Dataset:
    def __init__(self, ds_name):
        opts = dataset_params[ds_name]

        self.ds_name = ds_name
        self.task = opts.task if ds_name in ['ACSIncome', 'ACSEmployment'] else None
        self.columns = opts.columns
        self.train_cols = [col for col in self.columns if col != opts.label]
        self.label = opts.label
        self.sensitive_attributes = opts.sensitive_attributes
        self.use_sensitive = opts.use_sensitive
        self.categorical_columns = opts.categorical_columns
        self.state = opts.state if ds_name in ['ACSIncome', 'ACSEmployment'] else None 
        self.path = opts.path if ds_name == 'COMPAS' else None
    
        if self.use_sensitive:
            self.train_cols += [i for i in self.sensitive_attributes if i not in self.train_cols]
        else:
            self.train_cols = [i for i in self.train_cols if i not in self.sensitive_attributes]
    
    def get_folktables_data_source(self):
        data_source = ACSDataSource(survey_year=2018, horizon='1-Year', survey='person')
        acs_data = data_source.get_data(states=self.state, download=True)
        return self.task.df_to_pandas(acs_data, categories=self.categorical_columns, dummies=True)
    
    def _split_and_scale(self, features, labels, group, seed):
        features = features.fillna(features.median())
        features = features.infer_objects(copy=False)
        labels = np.squeeze(labels)
        group = np.squeeze(group)

        X_train_df, X_rem, y_train, y_rem, s_train, s_rem = train_test_split(
            features, labels, group, train_size=0.33, random_state=seed, stratify=labels
        )
        
        X_test_df, X_explain_df, y_test, y_explain, s_test, s_explain = train_test_split(
            X_rem, y_rem, s_rem, test_size=0.5, random_state=seed, stratify=y_rem
        )

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        y_explain = y_explain.astype(int)
        
        if len(np.unique(y_train)) > 1:
            ros = RandomOverSampler(random_state=seed)
            X_train_resampled, y_train_resampled = ros.fit_resample(X_train_df, y_train)
            
            indices_resampled = ros.sample_indices_
            s_train_resampled = pd.DataFrame(s_train).iloc[indices_resampled]

            X_train_df = pd.DataFrame(X_train_resampled, columns=X_train_df.columns)
            y_train = y_train_resampled
            s_train = s_train_resampled
        else:
            print(f"Warning: y_train has only one class: {np.unique(y_train)}. Skipping oversampling.")
        
        if self.ds_name in ['ACSIncome', 'ACSEmployment']:
            X_train  = X_train_df.to_numpy().astype(int)
            X_test = X_test_df.to_numpy().astype(int)
            X_explain = X_explain_df.to_numpy().astype(int)
            
            X_train_df = pd.DataFrame(X_train, columns=X_train_df.columns)
            X_test_df = pd.DataFrame(X_test, columns=X_test_df.columns)
            X_explain_df = pd.DataFrame(X_explain, columns=X_explain_df.columns)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df.values)
        X_test = scaler.transform(X_test_df.values)
        
        X_explain = X_explain_df.values

        return X_train_df, X_train, X_test_df, X_test, X_explain_df, X_explain, y_train, y_test, y_explain, s_train, s_test, s_explain

    def get_data(self, seed):
        if self.ds_name in ['ACSIncome', 'ACSEmployment']:
            features, labels, group = self.get_folktables_data_source()
            if self.ds_name in ['ACSIncome']:
                features['SEX'] = features['SEX'].replace({1: 0, 2: 1}).infer_objects(copy=False)
            elif self.ds_name == 'ACSEmployment':
                features['AGEP'] = np.where(features['AGEP'] <= 44, 0, 1)
            group = np.where(group == 1, 0, 1)
                                  
        elif self.ds_name == 'COMPAS':
            df = pd.read_csv(self.path)
            cat_cols_all = self.categorical_columns + [
                i for i in self.sensitive_attributes if isinstance(df[i].iloc[0], str)
            ]
            for col in cat_cols_all:
                df[col] = OrdinalEncoder().fit_transform(df[col].values.reshape(-1, 1))
            features = df[self.train_cols]
            labels = df[self.label].values
            group = df['race']
            
        features = features.infer_objects(copy=False)
        
        return self._split_and_scale(features, labels, group, seed)
