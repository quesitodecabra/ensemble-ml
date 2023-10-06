#!/usr/bin/env python
# coding: utf-8

# ## **1. Import Libraries**

# In[ ]:
import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Import the music and mental health data 
data = pd.read_csv('mxmh_survey_results 2.csv')



# - We see that the BPM data is written scientific notations, which is very weird

# ## We use numpy to calculate mathematical statistics for the dataset

# In[ ]:


# Filter the numeric columns
numeric_columns = data.select_dtypes(include=[np.number])

# Calculate descriptive statistics for the numeric columns
for feature in numeric_columns.columns:
    # Calculate the mean
    mean = np.mean(data[feature])
    # Calculate the median
    median = np.median(data[feature])
    # Calculate the variance
    variance = np.var(data[feature])
    # Calculate the standard deviation
    std_dev = np.std(data[feature])


# Get the non numeric columns
non_numeric_columns = data.select_dtypes(include=['object'])
non_numeric_columns.mode()


# Handle missing values in the dataset

age_median = data['Age'].median()
BPM_median = data['BPM'].median()


# Impute missing values in columns with median
data['Age'].fillna(age_median, inplace=True)
data['BPM'].fillna(BPM_median, inplace=True)


while_working_mode = data['While working'].mode()[0]
music_effects_mode = data["Music effects"].mode()[0]


# Impute missing values in the columns above with mode
data['While working'].fillna(while_working_mode, inplace=True)
data["Music effects"].fillna(music_effects_mode, inplace=True)

missing_values = data.isnull().sum()
print(missing_values)


numeric_columns = ['Age','Hours per day', 'BPM', 'Anxiety','Depression','Insomnia','OCD']
# Convert numeric columns to integer data type
data[numeric_columns] = data[numeric_columns].astype(int)

# Function to calculate outliers
def calculate_outliers(data):
    
    data = data.dropna()

    # First Quartile Value 
    q1 = np.percentile(data, 25)
    # Third Quartile Value of 
    q3 = np.percentile(data, 75)

    # Calculate Inter-Quartile Range
    iqr = q3 - q1

    # Calculate lower and upper bound
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    # Calculate outliers
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers


# Select numbered columns in the dataset
numeric_columns = ['Age', 'Hours per day', 'BPM', 'Anxiety', 'Depression', 'Insomnia', 'OCD']

# Calculate the outliers of these numeric columns
outliers = data[numeric_columns].apply(calculate_outliers, axis=0)
print(outliers)

bpm_outliers = calculate_outliers(data['BPM'])
print(bpm_outliers)
bpm_to_replace = bpm_to_replace = [ 0.0, 20.0, 4.0, 0.0, 8.0, 999999999.0, 624.0]

# Median value for BPM column
filtered_bpm = data[(data['BPM'] >= 60) & (data['BPM'] <= 300)]
median_bpm = filtered_bpm['BPM'].median()


# Replace missing BPM values with the median BPM
data['BPM'].fillna(median_bpm, inplace=True)


missing_values = data.isnull().sum()
print(missing_values)


# Identify 'BPM' values below 60
bpm_below_60_indices = data[data['BPM'] < 60].index

# Replace 'BPM' values below 60 with the median of the correct range
data.loc[bpm_below_60_indices, 'BPM'] = median_bpm




data.rename(columns={"Frequency [Pop]": "Frequency Pop"}, inplace=True)
data.rename(columns={"Frequency [Rock]": "Frequency Rock"}, inplace=True)
data.rename(columns={"Frequency [Metal]": "Frequency Metal"}, inplace=True)

# Perform ordinal encodings for Frequency Pop, Frequency Rock, Frequency Metal
# Frequency Pop: ['Very frequently', 'Sometimes', 'Rarely', 'Never']
ordinal_mapping = {
    'Never': 0,
    'Rarely': 1,
    'Sometimes': 2,
    'Very frequently': 3
}
# Apply mapping to data
data['Frequency Pop'] = data['Frequency Pop'].map(ordinal_mapping)
data['Frequency Rock'] = data['Frequency Rock'].map(ordinal_mapping)
data['Frequency Metal'] = data['Frequency Metal'].map(ordinal_mapping)

print(data['Frequency Pop'].unique())
print(data['Frequency Rock'].unique())
print(data['Frequency Metal'].unique())


import pandas as pd


# Define X and y for Objective 1
X_obj1 = data[['Age','Hours per day', 'Depression','OCD','Anxiety','BPM']]
y_obj1 = data[['Insomnia']]

# Define X and y for Objective 2
X_obj2 = data[['OCD','Insomnia','Depression','Frequency Pop','Frequency Rock','Frequency Metal']]
y_obj2 = data[['Anxiety']]




# In[ ]:


# # Split the dataset into training and validation sets for each objective.

# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

# Split the dataset for Objective 1 into training and validation sets
X_train_obj1, X_val_obj1, y_train_obj1, y_val_obj1 = train_test_split(X_obj1, y_obj1, test_size=0.2, random_state=42)

# Split the dataset for Objective 2 into training and validation sets
X_train_obj2, X_val_obj2, y_train_obj2, y_val_obj2 = train_test_split(X_obj2, y_obj2, test_size=0.2, random_state=42)


# Turn it into values arrays
y_train_obj1 = y_train_obj1.values.reshape(-1)
y_val_obj1 = y_val_obj1.values.reshape(-1)
y_train_obj2 = y_train_obj2.values.reshape(-1)
y_val_obj2 = y_val_obj2.values.reshape(-1)

# In[ ]:


# converting the y variable into a one-dimensional array using numpy.ravel(). The ravel() function returns a flattened one-dimensional array
# y_train_obj1 = np.ravel(y_train_obj1)
# y_train_obj2 = np.ravel(y_train_obj2)
import sklearn

# # Import the necessary libraries for the base models and ensemble method.
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# Base models for Objective 1
model1 = RandomForestRegressor()
model2 = GradientBoostingRegressor()
model3 = svm.SVR()
model4 = MLPRegressor(max_iter=1000)  # Replace CatBoost with neural network

# Base models for Objective 2
model5 = RandomForestRegressor()
model6 = GradientBoostingRegressor()
model7 = svm.SVR()
model8 = MLPRegressor(max_iter=1000)  # Replace CatBoost with neural network



from sklearn.model_selection import GridSearchCV


# ## RandomForestRegressor - Key hyperparameters to tune might include:
# - n_estimators
# - max_depth
# - min_samples_split
# 


param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}


# ## GradientBoostingRegressor - Key hyperparameters to tune might include:
# - n_estimators
# - learning rate
# - max_depth


param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [1, 3, 5]
}


# ## svm.SVR - Key hyperparameters to tune might include:
# - C
# - epsilon
# - kernel

# In[ ]:


param_grid_svr = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1],
    'kernel': ['linear', 'poly', 'rbf']
}


# ## MLPRegressor - Key hyperparameters to tune might include:
# - hidden_layer_sizes
# - activation
# - learning_rate
# 

# In[ ]:


param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}


# ## Running gridsearch for each model

# In[ ]:


grid_search_rf = GridSearchCV(estimator=model1, param_grid=param_grid_rf, cv=5)
grid_search_gb = GridSearchCV(estimator=model2, param_grid=param_grid_gb, cv=5)
grid_search_svr = GridSearchCV(estimator=model3, param_grid=param_grid_svr, cv=5)
grid_search_mlp = GridSearchCV(estimator=model4, param_grid=param_grid_mlp, cv=5)

# Fit the models
grid_search_rf.fit(X_train_obj1, y_train_obj1)
grid_search_gb.fit(X_train_obj1, y_train_obj1)
grid_search_svr.fit(X_train_obj1, y_train_obj1)
grid_search_mlp.fit(X_train_obj1, y_train_obj1)

# Print the best parameters
print("Best parameters for RandomForestRegressor: ", grid_search_rf.best_params_)
print("Best parameters for GradientBoostingRegressor: ", grid_search_gb.best_params_)
print("Best parameters for svm.SVR: ", grid_search_svr.best_params_)
print("Best parameters for MLPRegressor: ", grid_search_mlp.best_params_)


# In[ ]:

# Re-define models with the best parameters for Objective 1
model1 = RandomForestRegressor(max_depth=5, min_samples_split=10, n_estimators=50)
model2 = GradientBoostingRegressor(learning_rate=0.1, max_depth=1, n_estimators=100)
model3 = svm.SVR(C=10, epsilon=0.1, kernel='rbf')
model4 = MLPRegressor(activation='logistic', hidden_layer_sizes=(50, 50), learning_rate='constant',max_iter=5000)

# Re-define models with the best parameters for Objective 2
# Note: For simplicity, I'm assuming the best parameters are the same as for Objective 1. 
# You should replace these with the actual best parameters for Objective 2.
model5 = RandomForestRegressor(max_depth=5, min_samples_split=10, n_estimators=50)
model6 = GradientBoostingRegressor(learning_rate=0.1, max_depth=1, n_estimators=100)
model7 = svm.SVR(C=10, epsilon=0.1, kernel='rbf')
model8 = MLPRegressor(activation='logistic', hidden_layer_sizes=(50, 50), learning_rate='constant',max_iter=5000)


# # Train each base model on different subsets of the training set using bootstrap sampling for each objective.
# 

# In[ ]:


# y_val_obj1 = np.ravel(y_val_obj1)
# y_val_obj2 = np.ravel(y_val_obj2)


# In[ ]:


# Reshape the target variables for Objective 1 and Objective 2
 #y_train_obj1_1d = y_train_obj1.values.flatten()
 #y_train_obj2_1d = y_train_obj2.values.flatten()
 #print(y_train_obj1.shape)
 #print(y_train_obj2.shape)



# Train base models for Objective 1
base_models_obj1 = [model1, model2, model3, model4]
trained_models_obj1 = []

for model in base_models_obj1:
    subset_X_train_obj1, subset_y_train_obj1 = resample(X_train_obj1, y_train_obj1.flatten(), n_samples=len(X_train_obj1), replace=True, random_state=42)
    model.fit(subset_X_train_obj1, subset_y_train_obj1)
    trained_models_obj1.append(model)


# In[ ]:


# Train base models for Objective 2
base_models_obj2 = [model5, model6, model7, model8]
trained_models_obj2 = []

for model in base_models_obj2:
    subset_X_train_obj2, subset_y_train_obj2 = resample(X_train_obj2, y_train_obj2.flatten(), n_samples=len(X_train_obj2), replace=True, random_state=42)
    model.fit(subset_X_train_obj2, subset_y_train_obj2)
    trained_models_obj2.append(model)


# # Make predictions using each trained base model on the validation set for each objective.
# 

# In[ ]:


# Make predictions for each base model on the validation set for Objective 1
predictions_obj1 = []

for model in trained_models_obj1:
    y_pred_obj1 = model.predict(X_val_obj1)
    predictions_obj1.append(y_pred_obj1)


# In[ ]:


print(predictions_obj1)


# In[ ]:


# Make predictions for each base model on the validation set for Objective 2
predictions_obj2 = []

for model in trained_models_obj2:
    y_pred_obj2 = model.predict(X_val_obj2)
    predictions_obj2.append(y_pred_obj2)


# In[ ]:


print(predictions_obj2)


# # Combine the predictions from all base models using averaging for each objective.
# 

# In[ ]:


# Combine predictions using averaging for Objective 1
ensemble_predictions_obj1 = np.mean(predictions_obj1, axis=0)


# In[ ]:


print(ensemble_predictions_obj1)


# In[ ]:


# Combine predictions using averaging for Objective 2
ensemble_predictions_obj2 = np.mean(predictions_obj2, axis=0)


# In[ ]:


print(ensemble_predictions_obj2)







print(sklearn.__version__)