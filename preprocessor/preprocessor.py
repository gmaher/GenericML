import pandas as pd
import datetime as dt
import numbers

REDUCTION_CUTOFF = 0.3

class PreprocessorException(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return self.value

def max_date(x):
    d = dt.date(2020,1,1)
    if x > d:
        return d
    return x

def valid_string(x):
    if not type(x) == str:
        return False
    return True

def valid_date(x):
    if not isinstance(x,dt.date):
        return False
    return True

def valid_number(x):
    if not isinstance(x,numbers.Number):
        return False
    return True

def to_day(x):
    return x.day
def to_month(x):
    return x.month

class Preprocessor:
    def __call__(self,data):
        return data

class DataframePreprocessor(Preprocessor):
    def __init__(self, required_columns, label_column,string_columns=None, date_columns=None):
        self.required_columns = required_columns
        self.string_columns   = string_columns
        self.date_columns     = date_columns
        self.label_column     = label_column
        self.number_columns   = [c for c in self.required_columns if\
         not (any([c==k for k in self.string_columns]) or\
          any([c==h for h in self.date_columns]))]
    def preprocess(self,data):

        df = data
        for c in self.string_columns:
            df = df[df[c].map(valid_string)]

        for c in self.date_columns:
            df = df[df[c].map(valid_date)]

        for c in self.number_columns:
            df = df[df[c].map(valid_number)]

        if (1.0*df.shape[0])/data.shape[0] < REDUCTION_CUTOFF:
            raise PreprocessorException("After discarding malformed rows only {}\
            fraction of data remains".format(REDUCTION_CUTOFF))

        if any([c==self.label_column[0] for c in df.columns]):
            df_proc = df.loc[:,self.required_columns+self.label_column]
        else:
            df_proc = df.loc[:,self.required_columns]
        if not self.string_columns == None:
            df_proc[self.string_columns] = df_proc[self.string_columns].astype(str)

        if not self.date_columns == None:
            for c in self.date_columns:
                df_proc[c] = df_proc[c].apply(max_date)
                df_proc[c+'_DAY'] = df_proc[c].map(to_day)
                df_proc[c+'_MONTH'] = df_proc[c].map(to_month)

                df_proc[c+'_DAY'] = df_proc[c+'_DAY'].astype(int)

                df_proc[c+'_MONTH'] = df_proc[c+'_MONTH'].astype(int)
            df_proc = df_proc.drop(self.date_columns,axis=1)

        for c in self.number_columns:

            df_proc[c] = df_proc[c].fillna(df_proc[c].mean())

        df_proc[self.label_column] =\
            df_proc[self.label_column].fillna(df_proc[self.label_column].mean())

        df_dummies = pd.get_dummies(df_proc[self.string_columns])

        df_proc = df_proc.drop(self.string_columns,axis=1)

        df_proc = pd.merge(df_proc,df_dummies,left_index=True,right_index=True)

        return df_proc

    def get_columns(self,data):
        df = self.preprocess(data)
        self.all_columns = [c for c in df.columns if not c == self.label_column[0]]

    def predict(self,data):
        df = data
        if type(data) == dict:
            df = pd.DataFrame(columns=data.keys())
            df = df.append(data,ignore_index=True)

        df = self.preprocess(df)
        if any([c==self.label_column[0] for c in df.columns]):
            df = df.drop(self.label_column,axis=1)
        df = pd.DataFrame(columns=self.all_columns).append(df,ignore_index=True)
        df = df[self.all_columns]
        df = df.fillna(0)
        return df

    def train(self,data):
        df = self.preprocess(data)
        return df.drop(self.label_column,axis=1),df[self.label_column]
