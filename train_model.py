import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.impute import SimpleImputer
import joblib
import os

df = pd.read_csv('chicago_house.csv')

X = df[['Bedroom', 'Space', 'Room', 'Lot', 'Tax', 'Bathroom', 'Garage', 'Condition']]
y = df['Price']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25), 
    max_iter=2000, 
    solver='adam',  
    early_stopping=True,  
    random_state=42
)
mlp_model.fit(X_train, y_train)


estimators = [
    ('lr', lr_model),
    ('ridge', ridge_model),
    ('mlp', mlp_model)
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge()
)
stacking_model.fit(X_train, y_train)

# Save the models
if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(lr_model, 'models/linear_regression_model.joblib')
joblib.dump(ridge_model, 'models/ridge_regression_model.joblib')
joblib.dump(mlp_model, 'models/mlp_regressor_model.joblib')
joblib.dump(stacking_model, 'models/stacking_regressor_model.joblib')