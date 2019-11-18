# This example describes the validation method of the models

import pandas as pd
from sklearn.tree import  DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Data file to be considered
data_file = 'melb_data.csv'

# Data to be considered
data = pd.read_csv(data_file).dropna(axis=0)

# Sample Data columns for which data to be considered
sample_data_columns = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', 'Price']

# Print Sample Data
print("\n  Sample Data :: \n ", data[sample_data_columns].head())


# define the target
y = data.Price

# sample features to be considered
sample_data_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# Sample Data with defined Features
X = data[sample_data_features]


# Define Model
melbourne_model = DecisionTreeRegressor(random_state=1)

# Split the train data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# fit the model
melbourne_model.fit(train_X, train_y)

# get the predicted values
val_predicted = melbourne_model.predict(val_X)

print("\n Absolute Mean Error ::  ", mean_absolute_error(val_y, val_predicted))





