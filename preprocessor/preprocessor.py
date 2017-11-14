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
    def __init__(self, required_columns, label_column, string_columns=None, date_columns=None, date_parser=None):
        """
        Notes:
        - self.number columns will not contain self.label_column
        """
        self.required_columns = required_columns
        self.string_columns   = string_columns
        self.date_columns     = date_columns
        self.label_column     = label_column
        self.date_parser      = date_parser
        if (not self.date_parser == None) and (self.date_columns == None):
            raise RuntimeError("date_columns and date_parser must both be specified,\
             can't have one without the other")

        if (not self.string_columns==None) and (not self.date_columns == None):
            self.number_columns   = [c for c in self.required_columns if\
             not (any([c==k for k in self.string_columns]) or\
              any([c==h for h in self.date_columns]))]

        elif not self.string_columns == None:
            self.number_columns   = [c for c in self.required_columns if\
             not (any([c==k for k in self.string_columns]))]

        elif not self.date_columns == None:
            self.number_columns   = [c for c in self.required_columns if\
             not (any([c==k for k in self.date_columns]))]

        else:
            self.number_columns = None

    def extract_relevant_columns(self,df):
        if any([c == self.label_column[0] for c in df.columns]):
            df = df[self.required_columns+self.label_column]
        else:
            df = df[self.required_columns]

        return df

    def process_label_column(self,df):
        if any([c == self.label_column for c in df.columns]):
            df[self.label_column] =\
             df[self.label_column].fillna(df[self.label_column].mean())
            df = df[df[self.label_column].map(valid_number)]

        return df

    def process_date_columns(self,df):

        if not self.date_columns == None:
            for c in self.date_columns:
                df[c] = df[c].apply(max_date)
                df[c+'_DAY'] = df[c].map(to_day)
                df[c+'_MONTH'] = df[c].map(to_month)

                df[c+'_DAY'] = df[c+'_DAY'].astype(int)

                df[c+'_MONTH'] = df[c+'_MONTH'].astype(int)
            df = df.drop(self.date_columns,axis=1)
        return df

    def process_string_columns(self,df):
        if not self.string_columns == None:

            df[self.string_columns] = df[self.string_columns].astype(str)

            df_dummies = pd.get_dummies(df[self.string_columns])

            df = df.drop(self.string_columns,axis=1)

            df = pd.merge(df,df_dummies,left_index=True,right_index=True)

        return df

    def remove_invalid_rows(self,df):
        if not self.string_columns == None:
            for c in self.string_columns:
                df = df[df[c].map(valid_string)]

        if not self.date_columns == None:
            for c in self.date_columns:
                df[c] = df[c].map(self.date_parser)
                df = df[df[c].map(valid_date)]

        if not self.number_columns == None:
            for c in self.number_columns:
                df[c] = df[c].fillna(df[c].mean())
                df = df[df[c].map(valid_number)]

        return df

    def preprocess(self,data):

        df = data
        df = self.extract_relevant_columns(df)
        df = self.remove_invalid_rows(df)

        if (1.0*df.shape[0])/data.shape[0] < REDUCTION_CUTOFF:
            raise PreprocessorException("After discarding malformed rows only {}\
            fraction of data remains".format(REDUCTION_CUTOFF))

        df = self.process_label_column(df)
        df = self.process_string_columns(df)
        df = self.process_date_columns(df)

        return df

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
