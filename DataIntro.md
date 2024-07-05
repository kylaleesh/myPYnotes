###### Alt+Shift+v

# Important libraries
```py
!import pandas as pd
!import numpy as np
```

## Visualizations
```py
!import matplotlib.pyplot as plt
!import seaborn as sns
```

## Modeling
```py
!import sklearn
```

# PANDAS 

## Settings
showing all cols in tables ==> pd.set_option('display.max_columns', None) 

## Reading Data
```py
# read csv (default import row: header = 0):
data = pd.read_csv("Data/Energy_and_Water_Data_Disclosure.csv")
# read excel
df = pd.read_excel('./data/stock_data.xlsx',"Sheet1")
```

## viewing data & descriptive stats
| function | code |
| ------------- | ------------ |
| view first 5 lines | data.head(5) |
| view last 5 lines  | data.tail(5) |
| data shape: tuple (r,c)  | data.shape |
| basic info on data | data.info() |
| descriptive functions | data.ID.max() or min(), std(), mean(), median()|
| total numeric cols and concat str cols | data.sum() |
|summary of numeric columns| data.describe()|
|summary of object columns| data.describe(include = ['O'])|
|getting data columns | data.columns|
| data index description| data.index|
| count # of non-null cols | data.count() |
| count # of null cols |data.isna().sum() |

## Data cleaning
| Replacing "Not Available" with NaN | data.replace({"Not Available":np.nan}) |

## pd Series (s) & DataFrame (df)
```py
#s
s = pd.Series(data, index=index)
# Eg. you can specify index: pd.Series([9, 8, 7, 6], index=['*', '**', '***', '****'])
#series to DataFrame: 
s.to_frame()

#df
data = {
    'Country': ['US' , 'US', 'INDIA', 'INDIA'],
    'Year' : [2012,2013,2012,2013],
    'Population': [20,27,30,35]
}
df = pd.DataFrame(data)
```

## changing df
```py
#slice df using condition
df[
    df['Year'] > 2012
]
#create a new col named x
df['x'] = df['Population'] * 2

#transpose df
data.T

#sorting of df (inplace = True to modify original df directly)
data.sort_values(by='colname', ascending = False, inplace=True)
```
## Selecting data
by row number (index location - iloc):
* data.iloc[0] # first row of data frame - Note a Series data type output.
* data.iloc[1] # second row of data frame 
* data.iloc[-1] # last row of data frame 
* data.iloc[:,0] # first column of data frame 
* data.iloc[:,1] # second column of data frame 
* data.iloc[:,-1] # last column of data frame 

by data labels (.loc)
* data.loc[1:5,'SKU']


## RANDOM NOTES:
Directly create EDA reports using Pandas Profiling : https://github.com/pandas-profiling/pandas-profiling
```py
!pip install pandas_profiling
import pandas_profiling
pandas_profiling.ProfileReport(data)
```