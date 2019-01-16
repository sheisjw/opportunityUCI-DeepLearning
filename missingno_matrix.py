%matplotlib inline
import missingno as msno
import pandas as pd
import numpy as np
import matplotlib
from sklearn.preprocessing import Imputer

data_ori = pd.read_csv("data_raw_traininputs.csv")
#data_ori = data_ori.replace("NaN", np.nan)
#imputer = Imputer(missing_values=np.nan, strategy='median', axis=0)
#data_ori[['fnlwgt']] = imputer.fit_transform(data_ori[['fnlwgt']])
#census_data.isnull().sum()
msno.matrix(data_ori)

#print(data_ori.isnan().sum())
#msno.matrix(census_data)