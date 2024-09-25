import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle

data = pd.read_csv('realest.csv')

imputer = SimpleImputer(strategy='mean')
data_clean = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

X = data_clean[['Bedroom', 'Space', 'Room', 'Lot', 'Tax', 'Bathroom', 'Garage', 'Condition']]
y = data_clean['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

mlp_model = make_pipeline(StandardScaler(), MLPRegressor(max_iter=1000, random_state=42))
mlp_model.fit(X_train, y_train)

with open('linear_model.pkl', 'wb') as f:
    pickle.dump(linear_model, f)

with open('ridge_model.pkl', 'wb') as f:
    pickle.dump(ridge_model, f)

with open('mlp_model.pkl', 'wb') as f:
    pickle.dump(mlp_model, f)
