---
title: "Data Cleaning Project"
date: 2020-09-02
tags: [data wrangling, data science, data cleaning, messy data]
header:
  image: "/images/perceptron/header-image.png"
excerpt: "Data Wrangling, Data Science, Data Cleaning, Messy Data"
mathjax: "true"
---

# Data Cleaning


```python
# Importing libraries
import pandas as pd
import numpy as np

# Read csv file into a pandas dataframe
df = pd.read_csv(r'C:\Users\STEPHEN EDWIN\Documents\messydata.csv')

# Take a look at the first few rows
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PID</th>
      <th>ST_NUM</th>
      <th>ST_NAME</th>
      <th>OWN_OCCUPIED</th>
      <th>NUM_BEDROOMS</th>
      <th>NUM_BATH</th>
      <th>SQ_FT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001000.0</td>
      <td>104.0</td>
      <td>PUTNAM</td>
      <td>Y</td>
      <td>3</td>
      <td>1</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002000.0</td>
      <td>197.0</td>
      <td>LEXINGTON</td>
      <td>N</td>
      <td>3</td>
      <td>1.5</td>
      <td>--</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003000.0</td>
      <td>NaN</td>
      <td>LEXINGTON</td>
      <td>N</td>
      <td>NaN</td>
      <td>1</td>
      <td>850</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004000.0</td>
      <td>201.0</td>
      <td>BERKELEY</td>
      <td>12</td>
      <td>1</td>
      <td>NaN</td>
      <td>700</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>203.0</td>
      <td>BERKELEY</td>
      <td>Y</td>
      <td>3</td>
      <td>2</td>
      <td>1600</td>
    </tr>
  </tbody>
</table>
</div>



#### We will use a sample dataset that typically describes and highlights a lot of real world situations are typically experienced when working on data science projects.

### Regular Missing Values
#### By “regular missing values” I'm referring to the missing values that Pandas can detect. Let’s take a look at the “Street Number” column in our dataset.


```python
# Looking at the ST_NUM column
df['ST_NUM']
```




    0    104.0
    1    197.0
    2      NaN
    3    201.0
    4    203.0
    5    207.0
    6      NaN
    7    213.0
    8    215.0
    Name: ST_NUM, dtype: float64




```python
df['ST_NUM'].isnull()
```




    0    False
    1    False
    2     True
    3    False
    4    False
    5    False
    6     True
    7    False
    8    False
    Name: ST_NUM, dtype: bool



### Non-Regular Missing Values
#### Sometimes it might be the case where there’s missing values that have different formats. If we look at the “Number of Bedrooms” and "SQ_FT"columns, we see what I'm referring to.


```python
# Looking at the NUM_BEDROOMS column
df['NUM_BEDROOMS']
```




    0      3
    1      3
    2    NaN
    3      1
    4      3
    5    NaN
    6      2
    7      1
    8     na
    Name: NUM_BEDROOMS, dtype: object




```python
df['NUM_BEDROOMS'].isnull()
```




    0    False
    1    False
    2     True
    3    False
    4    False
    5     True
    6    False
    7    False
    8    False
    Name: NUM_BEDROOMS, dtype: bool




```python
# Looking at the SQ_FT column
df['SQ_FT']
```




    0    1000
    1      --
    2     850
    3     700
    4    1600
    5     800
    6     950
    7     NaN
    8    1800
    Name: SQ_FT, dtype: object




```python
df['SQ_FT'].isnull()
```




    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6    False
    7     True
    8    False
    Name: SQ_FT, dtype: bool



#### We can observe that Pandas did not recognize “na” and "--" as missing values. Unfortunately, the other types weren’t recognized. One way to handle this is to put them in a list. Then when we import the data and then Pandas will recognize them.


```python
# Making a list of missing value types
missing_values = ["n/a", "na", "--"]
df = pd.read_csv(r'C:\Users\STEPHEN EDWIN\Documents\messydata.csv', na_values = missing_values)
```


```python
# Now looking at the NUM_BEDROOMS column again
print (df['NUM_BEDROOMS'])
print (df['NUM_BEDROOMS'].isnull())
```

    0    3.0
    1    3.0
    2    NaN
    3    1.0
    4    3.0
    5    NaN
    6    2.0
    7    1.0
    8    NaN
    Name: NUM_BEDROOMS, dtype: float64
    0    False
    1    False
    2     True
    3    False
    4    False
    5     True
    6    False
    7    False
    8     True
    Name: NUM_BEDROOMS, dtype: bool



```python
# Now looking at the SQ_FT column again
print (df['SQ_FT'])
print (df['SQ_FT'].isnull())
```

    0    1000.0
    1       NaN
    2     850.0
    3     700.0
    4    1600.0
    5     800.0
    6     950.0
    7       NaN
    8    1800.0
    Name: SQ_FT, dtype: float64
    0    False
    1     True
    2    False
    3    False
    4    False
    5    False
    6    False
    7     True
    8    False
    Name: SQ_FT, dtype: bool


### Unexpected Missing Values
#### So far we’ve seen regular missing values, and non-regular missing values. What if we an unexpected type? For example, if our feature is expected to be a string, but there’s a numeric type and vice versa, then technically it is also a missing value. If we look at the “OWNER OCCUPIED” and "NUM BATH"columns, we can notice what I’m refering to.


```python
# Looking at the OWN_OCCUPIED column
print (df['OWN_OCCUPIED'])
print (df['OWN_OCCUPIED'].isnull())
```

    0      Y
    1      N
    2      N
    3     12
    4      Y
    5      Y
    6    NaN
    7      Y
    8      Y
    Name: OWN_OCCUPIED, dtype: object
    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    7    False
    8    False
    Name: OWN_OCCUPIED, dtype: bool



```python
# Detecting numbers
cnt=0
for row in df['OWN_OCCUPIED']:
    try:
        int(row)
        df.loc[cnt, 'OWN_OCCUPIED']=np.nan
    except ValueError:
        pass
    cnt+=1
```


```python
# Looking at the OWN_OCCUPIED column
print (df['NUM_BATH'])
print (df['NUM_BATH'].isnull())
```

    0         1
    1       1.5
    2         1
    3       NaN
    4         2
    5         1
    6    HURLEY
    7         1
    8         2
    Name: NUM_BATH, dtype: object
    0    False
    1    False
    2    False
    3     True
    4    False
    5    False
    6    False
    7    False
    8    False
    Name: NUM_BATH, dtype: bool



```python
df['NUM_BATH'] = pd.to_numeric(df['NUM_BATH'], errors='coerce')
df['NUM_BATH']
```




    0    1.0
    1    1.5
    2    1.0
    3    NaN
    4    2.0
    5    1.0
    6    NaN
    7    1.0
    8    2.0
    Name: NUM_BATH, dtype: float64




```python
df['NUM_BATH'].round().astype('Int64')
```




    0      1
    1      2
    2      1
    3    NaN
    4      2
    5      1
    6    NaN
    7      1
    8      2
    Name: NUM_BATH, dtype: Int64



### Checking for Missing Values
#### After we’ve cleaned up the dataset properly, we will probably want to check for them. For instance, we might want to look at the total number of missing values for each feature


```python
# Total missing values for each feature
df.isnull().sum()
```




    PID             1
    ST_NUM          2
    ST_NAME         0
    OWN_OCCUPIED    2
    NUM_BEDROOMS    3
    NUM_BATH        2
    SQ_FT           2
    dtype: int64




```python
# Any missing values?
print (df.isnull().values.any())
```

    True



```python
# Total number of missing values
print (df.isnull().sum().sum())
```

    12


### Replacing the missing values
#### Often times you’ll have to figure out how you want to handle missing values. Sometimes you’ll simply want to delete those rows, other times you’ll replace them. This phase shouldn’t be taken lightly. We will try a number of approaches, like filling in with specific values, mean, mode and median.


```python
# Location based replacement to fill in missing values with specific values
df.loc[2,'ST_NUM'] = 125
df.loc[6,'ST_NUM'] = 210
```


```python
#Fill in PID column with specific values
df.loc[4,'PID'] = 100005000
```


```python
# Replace using median
median = df['NUM_BEDROOMS'].median()
df['NUM_BEDROOMS'].fillna(median, inplace=True)
```


```python
# Replace using mode
df['OWN_OCCUPIED'].fillna(df['OWN_OCCUPIED'].mode()[0], inplace=True)
```


```python
# Replace using median
#med = df['NUM_BATH'].mean()
#df['NUM_BATH'].fillna(median, inplace=True)
df['NUM_BATH'] = df['NUM_BATH'].fillna(df['NUM_BATH'].median())
```


```python
# Replace using mean
df['SQ_FT'] = df['SQ_FT'].fillna(value=df['SQ_FT'].mean())
df['SQ_FT'].round().astype('Int64')
```




    0    1000
    1    1100
    2     850
    3     700
    4    1600
    5     800
    6     950
    7    1100
    8    1800
    Name: SQ_FT, dtype: Int64




```python
# Now check if there are any missing values?
print (df.isnull().values.any())
```

    False


### Conclusion
#### Dealing with messy data is inevitable. Infact, according to [IBM Data Analytics](https://www.ibm.com/blogs/bluemix/2017/08/ibm-data-catalog-data-scientists-productivity/) you can expect to spend up to 80% of your time cleaning data. Data cleaning is just part of the process on a data science project.
