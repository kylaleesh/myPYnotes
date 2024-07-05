import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv(path)
        

#Split data into train and test
y = data['Price']
X = data.drop(columns = ['Price'])

#Initialise model and fit it
model = LinearRegression()
model.fit(X, y)