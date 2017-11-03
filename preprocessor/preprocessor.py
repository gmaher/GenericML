import pandas as pd
from datetime import datetime

def max_date(x):
    d = datetime(2020,1,1)
    if x > d:
        return d
    return x

class Preprocessor:
    def __call__(self,data):
        return data

class DataframePreprocessor(Preprocessor):
    def __init__(self, required_columns, label_column,string_columns=None, date_columns=None):
        self.required_columns = required_columns
        self.string_columns   = string_columns
        self.date_columns     = date_columns
        self.label_column     = label_column

    def preprocess(self,data):
        df_proc = data.loc[:,self.required_columns]

        if not self.string_columns == None:
            df_proc[self.string_cols] = df_proc[self.string_cols].astype(str)

        if not self.date_columns == None:
            for c in self.date_columns:
                df_proc[c] = pd.DatetimeIndex(df_proc[c])
                df_proc[c] = df_proc[c].apply(max_date)
                df_proc[c+'_DAY'] = pd.DatetimeIndex(df_proc[c]).day
                df_proc[c+'_MONTH'] = pd.DatetimeIndex(df_proc[c]).month

                df_proc[c+'_DAY'] = df_proc[c+'_DAY'].astype(int)

                df_proc[c+'_MONTH'] = df_proc[c+'_MONTH'].astype(str)
            df_proc = df_proc.drop(date_cols,axis=1)
            df_proc = df_proc.fillna(0)

        df_proc = pd.get_dummies(df_proc)
        return df_proc

    def predict(self,data):
        df = preprocess(data)
        return df.drop(self.label_column,axis=1)

    def train(self,data):
        df = preprocess(data)
        return df.drop(self.label_column,axis=1),df[self.label_column]
