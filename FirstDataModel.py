import pandas as pd
from sklearn.tree import  DecisionTreeRegressor


# Define the file path
melbourne_file_path = 'melb_data.csv'

# Define the Data Columns to be seen
melbourne_data_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', 'Price']

# Read Data from File
melbourne_data = pd.read_csv(melbourne_file_path)

# Print the file Data for the reference
print(" \n Melbourne Data columns :: \n", melbourne_data[melbourne_data_features])

# Define the Target
y = melbourne_data.Price

# Define the Melbourne Features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# print the features
print(" \n Melbourne Features :: \n ",  melbourne_features)

X = melbourne_data[melbourne_features]

# print melbourne features data
print("\n Melbourne Features  Data \n", X.head())

##
# Building my model in 3 Steps
# Step-1 Define the type of model ( Mathematical model to be selected from the available Packages )
# Step-2 Fit the model. Capture patterns provided by the data. This is heart of modelling
# Step-3 Predict
# Step-4 Evaluate
##

# Define model
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit the model
melbourne_model.fit(X, y)

print(" \n  Making prediction for Following 5 Houses :: ")
print("\n", X.head())
print("\n The predictions are ::")
print("\n Predictions are :::")
print(melbourne_model.predict(X.head()))











