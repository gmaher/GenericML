import sys
import os
import pandas as pd
from datetime import date
import random
sys.path.append(os.path.abspath('../'))
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from preprocessor.preprocessor import DataframePreprocessor
from model.model import SKLearnModel

def random_date(start,end):
    start_date = date.toordinal(start)
    end_date = date.toordinal(end)
    random_day = date.fromordinal(random.randint(start_date, end_date))
    return random_day

preprocessor = DataframePreprocessor(required_columns=['A','B','C'],
    string_columns=['B'],
    label_column=['D'],
    date_columns=['A'])

regressor = RandomForestRegressor()

model = SKLearnModel(regressor)
model.setPreprocessors(preprocessor.predict,preprocessor.train)
#############################
# Create fictional training dataset
############################
N = 100

start_date = date.fromordinal(1000)
end_date = date.fromordinal(2000000)

strings = ['cat','dog','horse','giraffe']

number_start = -100
number_end = 1e10

columns = ['A','B','C','D']
df = pd.DataFrame(columns=columns)

for i in range(N):
    d1 = random_date(start_date,end_date)
    d2 = random.choice(strings)
    d3 = np.random.rand()*(number_end-number_start) + number_start
    d4 = np.random.rand()*(number_end-number_start) + number_start

    r = np.random.rand()
    if r < 0.1:
        d1 = None
    if r > 0.9:
        d4 = None
    if r > 0.1 and r < 0.2:
        d3 = None
    if r > 0.2 and r < 0.3:
        d2 = None
    if r>0.3 and r < 0.4:
        d2 = np.random.rand()*(number_end-number_start) + number_start
        d3 = 'blah'
        d1 = -2
    if r>0.4 and r < 0.5:
        d1 = 'brrr'

    d = {"A":d1, "B":d2, "C":d3, "D":d4}

    df = df.append(d,ignore_index=True)
preprocessor.get_columns(df)
model.train(df)

X = model.predict(df)
