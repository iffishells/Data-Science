## importing all model and lib

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


file_path = "Datasets/train.csv"

home_data = pd.read_csv(file_path)

# home_data.head()  # head

rfr = RandomForestRegressor(random_state=1) # static data

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']


# print(features)

target_Value = home_data ["SalePrice"]

y = target_Value # as convention save in y

## for train x
## fitting the model 
X = home_data[features]
rfr.fit(X,y)  ## Fitting

## prediction 



# 
