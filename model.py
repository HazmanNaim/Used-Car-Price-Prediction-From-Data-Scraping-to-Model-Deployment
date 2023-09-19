# Import libraries and packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

# Import data
data = pd.read_csv('cleaned_data.csv')
included = ['Assembled','brand_tier', 'Engine CC', 'vehicle_age', 'price']
df = data[included]

cat_features = ['Assembled',
                'brand_tier']

# Assembled : 'Official Import', 'Parallel Import', 'Locally Built'
# brand_tier: 'S', 'A', 'B', 'C', 'D'

num_features = ['Engine CC',
                'vehicle_age']

# Engine CC: int64
# vehicle_age: int64

response = 'price'

# Create pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline

transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
                                              ('num', MinMaxScaler(), num_features)], remainder="passthrough")

# Model development
from sklearn.model_selection import train_test_split
import pickle

x_data = df.drop(response, axis=1)
y_data = df[response]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

# Train the Model
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(bootstrap=False,
                           max_depth=50,
                           max_features='log2',
                           min_samples_leaf=1,
                           min_samples_split=6)

pipe = Pipeline([
    ('ColumnTransformer', transformer),
    ('Model', rf_model)
])

pipe.fit(x_train,  y_train)

n = x_data.shape[0]
k = len(x_data.columns)

# Model Prediction
predictions_train = pipe.predict(x_train)
predictions_test = pipe.predict(x_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

# Model evaluation
def cross_fold(pipeline):
    Rcross = cross_val_score(pipeline, x_data, y_data, cv=10).mean()
    RMSEcross = np.sqrt(-cross_val_score(model,x_data, y_data, scoring="neg_mean_squared_error", cv=10)).mean()
    return Rcross, RMSEcross

def evaluation_train(y, predictions):
    r2_train = r2_score(y, predictions)
    return r2_train

def evaluation_test(y, predictions):
    rmse_test = np.sqrt(mean_squared_error(y, predictions))
    r2_test = r2_score(y, predictions)
    return rmse_test, r2_test

def R2_train_adjusted(r_squared,n,k):
    adjusted_r2 = r2_train-(k-1)/(n-k)*(1-r2_train)
    return adjusted_r2

def R2_test_adjusted(r_squared,n,k):
    adjusted_r2_train = r2_test-(k-1)/(n-k)*(1-r2_test)
    return adjusted_r2_train

# Model validation
r2_train = evaluation_train(y_train, predictions_train)
rmse_test, r2_test = evaluation_test(y_test, predictions_test)
adjusted_r2_train = R2_train_adjusted(r2_train,n,k)
adjusted_r2_test = R2_test_adjusted(r2_test,n,k)
Rcross = cross_val_score(pipe, x_data, y_data, cv=10).mean()
RMSEcross = np.sqrt(-cross_val_score(pipe,x_data, y_data, scoring="neg_mean_squared_error", cv=10)).mean()

print("-"*30)
print("RANDOM FOREST REGRESSOR") # Changeable
print("-"*30)
print("R2 Score (Train):", r2_train)
print("Adjusted R2 Score (Train):", adjusted_r2_train)
print("RMSE (Test):", rmse_test)
print("R2 Score (Test):", r2_test)
print("Adjusted R2 Score (Test):", adjusted_r2_test)
print("RMSE Cross-Validation:", RMSEcross)
print("R2 Score Cross-Validation:", Rcross)
print("-"*30)

# Save the model
import joblib
joblib.dump(pipe, 'model.pkl')
print("Model dumped!")

# Load the model that just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(x_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")