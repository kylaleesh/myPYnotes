# ML models - Supervised:
## Linear Regression
* Purpose: Predicting a continuous numerical value (dependent variable).
* Output: Continuous values.
* Example: Predicting house prices based on features like size, number of rooms, location, etc.

### Sklearn Package
#### Baseline model 
```py
# Importing libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split  

# Read data
data = pd.read_csv(path)

# Split data into train and test
y = data['Price']
X = data.drop(columns = ['Price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) # 30% for testing

# Initialise model and fit it
model = LinearRegression()
model.fit(X, y)

# Getting model coefficients and intercepts
intercept = model.intercept_
coeff = model.coef_

# Prediction
prediction = model.predict(X_test)
```
#### Refit model with scaled data 
Scaled data can be Standardisation, MinMaxScaling...
```py
# preprocess the data with scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1 - Standardisation
ss = StandardScaler(with_std = True)
X_std = ss.fit_transform(X)

# 2 - MinMaxScaler
min_max = sklearn.preprocessing.MinMaxScaler()
X_mm = min_max.fit_transform(X)
```
#### Check the scoring of the test data (R^2)
This indicates how well the regression predictions approximate the real data points, range from 0 (does not explain any variability) to 1 (perfectly explain).
```py
# R-squared value of train set
lm.score(X,y)
# R-squared value of test set
lm.score(X_test, y_test)
```

### statsmodels.api Package
#### Baseline Model
```py
import statsmodels.api as sm

# Fit and make predictions by the model (OLS Regression Model)
X = sm.add_constant(X)
model = sm.OLS(list(y), X).fit()
predictions = model.predict(X)

# Get model summary - R^2, adjusted R^2, F-stats, p-value and many more
model.summary()
```

#### Drop cols with p-value < 0.05 and refit & predict
```py
# drop col
columns =  ['Latitude','Longitude','Water Intensity (All Water Sources) (gal/ft²)', 'Number of Buildings - Self-reported','Weather Normalized Site Natural Gas Use (therms)',
          'Occupancy','DOF Gross Floor Area','Census Tract', 'Property Id', 'log_Water Intensity (All Water Sources) (gal/ft²)']
train_df2 = train_df1.drop(columns, axis = 1)

# retrain and refit
train_df2 = sm.add_constant(train_df2)
model_new = sm.OLS(list(y), train_df2).fit()
predictions = model_new.predict(train_df2)

# get second model summary stats
model_new.summary()
```
#### Check for heteroskedasticity
whether the IV affects the DVs to different extent
```py
import statsmodels.stats.api as sms
from statsmodels.compat import lzip

name = ['Lagrange', 'p-value', 'f-value', 'f p-value']
test = sms.het_breuschpagan(model_new.resid, model_new.model.exog)
lzip(name, test)
'''
[('Lagrange', 683.3447107640369),
 ('p-value', 1.6174569935774892e-138),
 ('f-value', 63.759993023071495),
 ('f p-value', 6.713480465829707e-147)] #significant
 '''
 # p-value if heteroskedasticity is significant
 # F-value to gauge the overall impact of heteroskedasticity
 ```

##### If there is heteroskedasticity, check for multicollinearity with VIF
After checking, drop the relevant columns and continue iterate through the model
```py
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_vif_(X, thresh=2.0):
    variables = list(range(X.shape[1]))
    drop_cols = list()
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            drop_cols.append(X.iloc[:, variables].columns[maxloc])
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables], drop_cols

X_train, drop_cols = calculate_vif_(train_df2)
```
### Evaluate Regression Model - metrics & cross_val_score
```py
# Calculating Regression Evaulation metrics
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y, predictions))
print('MSE:', metrics.mean_squared_error(y, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y, predictions)

# Estimating the accuracy of our model by splitting the data, fitting a model and computing the score 5 consecutive times (with different splits each time):
from sklearn.model_selection import cross_val_score
regr = linear_model.LinearRegression()
scores = cross_val_score(regr, train_df3, y, cv=5)
# The mean score and the 95% confidence interval of the score estimate are hence given by: cross validated accuracy.
print(np.mean(scores))
```