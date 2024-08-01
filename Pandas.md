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
df = pd.DataFrame(data, columns = ["student_id", "age","area"])
```

## changing df
```py
#slice df using condition
df[
    (condition1) & (condition2) | ~(condition3)
]
# ~ not
# & and
# | or

#create a new col named x
df['x'] = df['Population'] * 2

#transpose df
data.T

#sorting of df (inplace = True to modify original df directly)
data.sort_values(by='colname', ascending = False, inplace=True)
```
## Selecting data
by row number (index location - iloc):
* data.iloc[0] # first row of data frame
* data.iloc[1] # second row of data frame 
* data.iloc[-1] # last row of data frame 
* data.iloc[:,0] # first column of data frame 
* data.iloc[:,1] # second column of data frame 
* data.iloc[:,-1] # last column of data frame 

by data labels (.loc)
* data.loc[1:5,'SKU']

## Data Cleaning
### Drop duplicates
```py
DataFrame.drop_duplicates(subset=None, *, keep='first', inplace=False, ignore_index=False)
```
Parameters:
* subset: column label or sequence of labels, optional
    * Only consider certain columns for identifying duplicates, by default use all of the columns.
* keep{‘first’, ‘last’, False}, default ‘first’
Determines which duplicates (if any) to keep.
    * ‘first’ : Drop duplicates except for the first occurrence.
    * ‘last’ : Drop duplicates except for the last occurrence.
    *  False : Drop all duplicates.
* inplace: bool, default False
    * Whether to modify the DataFrame rather than creating a new one.
* ignore_index: bool, default False
    * If True, the resulting axis will be labeled 0, 1, …, n - 1.

### Finding Duplicates
```py
df.duplicated(subset=None, keep='first')
```
* subset: column label or sequence of labels, optional. Only consider certain columns for identifying duplicates, by default use all of the columns.
* keep: {‘first’, ‘last’, False}, default ‘first’. Determines which duplicates (if any) to mark.
    * first : Mark duplicates as True except for the first occurrence.
    * last : Mark duplicates as True except for the last occurrence.
    * False : Mark all duplicates as True.

### Missing values
```py
# Drop Missing Values
DataFrame.dropna(*, axis=0, how=_NoDefault.no_default, thresh=_NoDefault.no_default, subset=None, inplace=False, ignore_index=False)

# Fill Missing Values with 0
data["colname"].fillna(0, inplace=True) #specific col
data.fillna(0, inplace=True) #all df
```

### Removing/Drop columns
```py
DataFrame.drop(['list of col'], axis = 1)
```

### Renaming Columns
```py
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
df.rename(columns={"A": "a", "B": "c"}, inplace = True)
'''
   a  c
0  1  4
1  2  5
2  3  6
'''
df.rename(index={0: "x", 1: "y", 2: "z"}, inplace = True)
'''
   A  B
x  1  4
y  2  5
z  3  6
'''
```
### check if a value is in list
Whether elements in Series are contained in values.
```py
Series.isin(values)
```
Return a boolean Series showing whether each element in the Series matches an element in the passed sequence of values exactly.

### Creating new columns using assign
```py
DataFrame.assign(**kwargs)
df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32)
''' new temp_f col:
          temp_c  temp_f
Portland    17.0    62.6
Berkeley    25.0    77.0
'''
```
## Transform DataFrames
### Union 2 df
```py
pd.concat([df1,df2])
```
### Merge df
```py
left_df.merge(right_df, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=None, indicator=False, validate=None)
```
* how: Type of merge to be performed. {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}, default ‘inner’
    * left: use only keys from left frame, similar to a SQL left outer join; preserve key order.
    * right: use only keys from right frame, similar to a SQL right outer join; preserve key order.
    * outer: use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically.
    * inner: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys.
    * cross: creates the cartesian product from both frames, preserves the order of the left keys.
* on: label or list. Column or index level names to join on. These must be found in both DataFrames. If on is None and not merging on indexes then this defaults to the intersection of the columns in both DataFrames.
* left_onlabel or list, or array-like. Column or index level names to join on in the left DataFrame. Can also be an array or list of arrays of the length of the left DataFrame. These arrays are treated as if they are columns.
* right_onlabel or list, or array-like. Column or index level names to join on in the right DataFrame. Can also be an array or list of arrays of the length of the right DataFrame. These arrays are treated as if they are columns.
* left_indexbool, default False. Use the index from the left DataFrame as the join key(s). If it is a MultiIndex, the number of keys in the other DataFrame (either the index or a number of columns) must match the number of levels.
* right_indexbool, default False. Use the index from the right DataFrame as the join key. Same caveats as left_index.
* sortbool, default False. Sort the join keys lexicographically in the result DataFrame. If False, the order of the join keys depends on the join type (how keyword).
* suffixeslist-like, default is (“_x”, “_y”). A length-2 sequence where each element is optionally a string indicating the suffix to add to overlapping column names in left and right respectively. Pass a value of None instead of a string to indicate that the column name from left or right should be left as-is, with no suffix. At least one of the values must not be None.
* copybool, default True. If False, avoid copy if possible.

### GROUP BY - SELECT count(*) in sql
```py
# SQL 50 Q.1581
df[df.transaction_id.isna()].groupby(by = 'customer_id')['visit_id'].count().rename('count_no_trans').reset_index(name = 'renamecol')

# df[df.transaction_id.isna()]: Getting the null values
# .groupby(by = 'customer_id')['visit_id'].count(): counting the num of visits
# .rename('count_no_trans'): rename the col headers
# .reset_index(): bring back the 'customer_id' column
```

### GROUP BY - ranking 
```py
df = employees.groupby(['reports_to']).agg(
    reports_count = ('employee_id','count'),
    average_age = ('age', lambda x: (x.mean() + 1e-6).round() ),
).reset_index()
```

### GROUP BY - aggregating 
```py
df = employees.groupby(['reports_to']).agg(
    reports_count = ('employee_id','count'),
    average_age = ('age', lambda x: (x.mean() + 1e-6).round() ),
).reset_index()
```
Creating multiple aggregated data

### Using a function to transform WHOLE DataFrame
DataFrame.transform(func, axis=0, *args, **kwargs)[source]
```py
    df.transform(lambda x: x + 1) #adding 1 to the all rows and cols of df
    df.transform(func = ['sqrt', 'exp']) #getting sqrt and exp on all cols of df
    
    #creating a col called max salary
    df['max_salary'] = df.groupby('name_y') ['salary'].transform(max)
```

### Getting difference from the prev row
```py
DataFrame.diff(periods=1, axis=0)

df.datecol.diff() == '1 days' # date columns
weather.temperature.diff() > 0 # int columns
```
Calculates the difference of a DataFrame element compared with another element in the DataFrame (default is element in previous row).
* periods: int, default 1. Periods to shift for calculating difference, accepts negative values.
* axis: {0 or ‘index’, 1 or ‘columns’}, default 0. Take difference over rows (0) or columns (1).


### Pivot Tables
```py
df.pivot(index='indexcol', columns='colnames', values='numeric_col')
#eg
df.pivot(index='month', columns='city', values='temperature')
'''
From:
| city         | month    | temperature |
| ------------ | -------- | ----------- |
| Jacksonville | January  | 13          |
| Jacksonville | February | 23          |
| Jacksonville | March    | 38          |
| Jacksonville | April    | 5           |
| Jacksonville | May      | 34          |
| ElPaso       | January  | 20          |
| ElPaso       | February | 6           |
| ElPaso       | March    | 26          |
| ElPaso       | April    | 2           |
| ElPaso       | May      | 43          |

To:
| month    | ElPaso | Jacksonville |
| -------- | ------ | ------------ |
| April    | 2      | 5            |
| February | 6      | 23           |
| January  | 20     | 13           |
| March    | 26     | 38           |
| May      | 43     | 34           |
'''
```
### Melt Table
```py
df.melt(id_vars = 'col_index', var_name = 'variable', value_name = 'value')
'''
From:
| product     | quarter_1 | quarter_2 | quarter_3 | quarter_4 |
| ----------- | --------- | --------- | --------- | --------- |
| Umbrella    | 417       | 224       | 379       | 611       |
| SleepingBag | 800       | 936       | 93        | 875       |

To:
| product     | quarter (variable)  | sales (value) |
| ----------- | ------------------- | ------------- |
| Umbrella    | quarter_1           | 417           |
| SleepingBag | quarter_1           | 800           |
| Umbrella    | quarter_2           | 224           |
| SleepingBag | quarter_2           | 936           |
| Umbrella    | quarter_3           | 379           |
| SleepingBag | quarter_3           | 93            |
'''
```
### Apply() Function
 Find length of content which is greater than 15: 
 ```py
 df.colname.apply(len) > 15
 df[df.col.str.len() > 15]
 ```
The apply() function is particularly useful for performing element-wise operations on Series or DataFrame columns, especially when you want to transform or manipulate data using a custom function. It's often used to avoid explicit loops, which can be slower and less concise.

## Transform Strings
### Regex
https://www.w3schools.com/python/python_regex.asp
* []	A set of characters	"[a-m]"	
* \	Signals a special sequence (can also be used to escape special characters)	"\d"	
* .	Any character (except newline character)	"he..o"	
* ^	Starts with	"^hello"	
* \$	Ends with	"planet$"	
* \*	Zero or more occurrences	"he.*o"	
* \+	One or more occurrences	"he.+o"	
* ?	Zero or one occurrences	"he.?o"	
* {}	Exactly the specified number of occurrences	"he.{2}o"	
* |	Either or	"falls|stays"	
* ()	Capture and group	
```py
#checking if df data "starts with DIAB1" (^DIAB1) or contains " DIAB1"
patients[patients.conditions.str.contains(r'(^DIAB1)|( DIAB1)')]

#checking for emails with:
#The prefix name is a string that may contain letters (upper or lower case), digits, underscore '_', period '.', and/or dash '-'. The prefix name must start with a letter.
#The domain is '@leetcode.com'.
users[
        users.mail.str.match(r'^[A-Za-z][A-Za-z0-9_.-]*@leetcode[.]com$'')
    ]
```


## RANDOM NOTES:
### Rounding Even 0.5 Issues
```py
round(8.5) # 8
round(res.average_age + 10e-3) # to prevent rounding issues
```

### Pandas Profiling Reports
Directly create EDA reports using Pandas Profiling : https://github.com/pandas-profiling/pandas-profiling
```py
!pip install pandas_profiling
import pandas_profiling
pandas_profiling.ProfileReport(data)
```