import  pandas as pd
from sklearn.tree import DecisionTreeRegressor

train_file_path = 'train.csv'
train_data = pd.read_csv(train_file_path)


# print the list of columns of the train data
print(" \n Train Data columns :: \n", train_data.columns)


# print the relevant feature data
train_data_feature = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','SalePrice']
print(" \n Train Data  :: \n", train_data[train_data_feature])


# Select the Target Feature as SalePrice
y = train_data.SalePrice

# Define the features to be used in Modelling
train_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#Define the Training Data
X = train_data[train_features]

# Define the Model
train_model = DecisionTreeRegressor(random_state=1)

#Fit the Model
train_model.fit(X, y)


#Print the Predicton
print(" \n  Making prediction for Following 5 Houses :: ")
print("\n", X.head())
print("\n The predictions are ::")
print("\n Predictions are :::")
print(train_model.predict(X.head()))



