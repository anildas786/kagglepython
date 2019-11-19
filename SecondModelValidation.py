import pandas as pd
from sklearn.tree import  DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# set the Data file Path
data_file_path = 'train.csv'

# Read the Data file
read_data = pd.read_csv(data_file_path)

# Set the data columns to be considered to be path of Data.
data_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'SalePrice']

# Now you are ready with the filtered Data
data = read_data[data_columns]

# Print the Filtered Data
print(" Data :: ", data.head())

# Set the Target
y = data.SalePrice

# set the feature for the sample Data
data_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Set the Data for sampling
X = data [data_features]


# Split the Data with train and validation Data
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

# define the model
model = DecisionTreeRegressor(random_state=1)

# fit the model
model.fit(X_train, y_train)

# get the predicted value
y_predicted = model.predict(X_val)

# print the mean absolute error
print(" \n Mean absolute Error ::: \n ", mean_absolute_error(y_val, y_predicted))






