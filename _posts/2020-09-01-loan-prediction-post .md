---
title: "Loan Approval Project"
date: 2020-09-01
tags: [data wrangling, data science, loan approval prediction]
header:
  image: "/images/perceptron/display-page.jpg"
excerpt: "Data Wrangling, Data Science, Loan Approval Prediction"
mathjax: "true"
---

# Loan Approval Prediction


```python
# We import the necessary libraries
%pylab inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from sklearn.utils import resample
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
```

    Populating the interactive namespace from numpy and matplotlib




## The Dataset
#### The data used for this project is from a real credit union in a university, called projectdata.csv. The data has a shape of 593 x 12


```python
#reading the dataset in dataframe using pandas
df = pd.read_csv (r'C:\Users\STEPHEN EDWIN\Documents\projects\projectdata.csv')
```


```python
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
      <th>Form_No</th>
      <th>Gender</th>
      <th>Married</th>
      <th>Guarantor</th>
      <th>Guarantor_Contribution</th>
      <th>University_Staff</th>
      <th>Income_Level</th>
      <th>LoanAmount</th>
      <th>Credit_History</th>
      <th>Union_member</th>
      <th>pledge</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FN001</td>
      <td>Male</td>
      <td>NaN</td>
      <td>Valid</td>
      <td>Ok</td>
      <td>Yes</td>
      <td>150000</td>
      <td>100000.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FN002</td>
      <td>Male</td>
      <td>Yes</td>
      <td>Valid</td>
      <td>Ok</td>
      <td>Yes</td>
      <td>50000</td>
      <td>18000.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FN003</td>
      <td>Male</td>
      <td>Yes</td>
      <td>Valid</td>
      <td>Ok</td>
      <td>Yes</td>
      <td>80000</td>
      <td>50000.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FN004</td>
      <td>Male</td>
      <td>Yes</td>
      <td>Invalid</td>
      <td>nil</td>
      <td>Yes</td>
      <td>50000</td>
      <td>20000.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FN005</td>
      <td>Male</td>
      <td>No</td>
      <td>Valid</td>
      <td>Ok</td>
      <td>NaN</td>
      <td>45000</td>
      <td>10000.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (593, 12)




```python
#Next we get a summary of numerical variables
df.describe()
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
      <th>Income_Level</th>
      <th>LoanAmount</th>
      <th>Credit_History</th>
      <th>Union_member</th>
      <th>pledge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>593.000000</td>
      <td>581.000000</td>
      <td>593.000000</td>
      <td>593.000000</td>
      <td>593.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>73277.403035</td>
      <td>27332.185886</td>
      <td>0.733558</td>
      <td>0.391231</td>
      <td>0.409781</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47980.114779</td>
      <td>24958.789763</td>
      <td>0.442471</td>
      <td>0.488438</td>
      <td>0.492208</td>
    </tr>
    <tr>
      <th>min</th>
      <td>25000.000000</td>
      <td>5000.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>45000.000000</td>
      <td>11000.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>60000.000000</td>
      <td>17000.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>80000.000000</td>
      <td>30000.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>300000.000000</td>
      <td>150000.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Next we do some investigation into our data to get some understanding of how our data is, its' distribution and the relationships between the variables. This will enable us to make meaningful assumptions espcially when we will handle missing values (Nans) in our dataset.


```python
df['LoanAmount'].hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xf33b92a3c8>




![png](output_8_1.png)



```python
df.boxplot(column='LoanAmount')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xf33bff5518>




![png](output_9_1.png)



```python
# We check for missing values in the data
df.apply(lambda x: sum(x.isnull()),axis=0)
```




    Form_No                    1
    Gender                    13
    Married                    4
    Guarantor                  2
    Guarantor_Contribution     0
    University_Staff          28
    Income_Level               0
    LoanAmount                12
    Credit_History             0
    Union_member               0
    pledge                     0
    Loan_Status                0
    dtype: int64



#### Now inspect those with missing values and do a value count for each


```python
df['Gender'].value_counts()
```




    Male      447
    Female    133
    Name: Gender, dtype: int64




```python
df['Married'].value_counts()
```




    Yes    383
    No     206
    Name: Married, dtype: int64




```python
df['Guarantor'].value_counts()
```




    Valid      454
    Invalid    137
    Name: Guarantor, dtype: int64




```python
df['University_Staff'].value_counts()
```




    Yes    378
    No     187
    Name: University_Staff, dtype: int64




```python
table = df.pivot_table(values='LoanAmount', index='Guarantor' ,columns='Credit_History', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Guarantor'],x['Credit_History']]
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
df['University_Staff'].fillna('Yes',inplace=True)
df['Married'].fillna('Yes',inplace=True)
df['Gender'].fillna('Male',inplace=True)
df['Guarantor'].fillna('Valid',inplace=True)
```


```python
#check for missing to verify that there are no more Nans in our data
df.apply(lambda x: sum(x.isnull()),axis=0)
```




    Form_No                   1
    Gender                    0
    Married                   0
    Guarantor                 0
    Guarantor_Contribution    0
    University_Staff          0
    Income_Level              0
    LoanAmount                0
    Credit_History            0
    Union_member              0
    pledge                    0
    Loan_Status               0
    dtype: int64



#### It is important to note here that the method adopted for encoding is very important so that data is not misrepresented or priority given to one class over the other. In our case, we can safely use LabelEncoder since our categorical variable will be properly represented when encoded.


```python
#LabelEncoding used to convert all categorical values into numeric
var_mod = ['Gender','Married','Guarantor', 'Guarantor_Contribution', 'Loan_Status','University_Staff']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes
```




    Form_No                    object
    Gender                      int32
    Married                     int32
    Guarantor                   int32
    Guarantor_Contribution      int32
    University_Staff            int32
    Income_Level                int64
    LoanAmount                float64
    Credit_History              int64
    Union_member                int64
    pledge                      int64
    Loan_Status                 int32
    dtype: object




```python
#Here I choose to type cast all into int64
df['LoanAmount']=df['LoanAmount'].astype(np.int64)
df['Gender'] = df['Gender'].astype(np.int64)
df['Married']=df['Married'].astype(np.int64)
df['University_Staff']=df['University_Staff'].astype(np.int64)
df['Guarantor_Contribution']=df['Guarantor_Contribution'].astype(np.int64)
df['Loan_Status']=df['Loan_Status'].astype(np.int64)
df['Guarantor']=df['Guarantor'].astype(np.int64)
df.dtypes
```




    Form_No                   object
    Gender                     int64
    Married                    int64
    Guarantor                  int64
    Guarantor_Contribution     int64
    University_Staff           int64
    Income_Level               int64
    LoanAmount                 int64
    Credit_History             int64
    Union_member               int64
    pledge                     int64
    Loan_Status                int64
    dtype: object




```python
#We now drop the Form_No axis, since it doesn't contain any useful information
df = df.drop('Form_No', axis = 1)
y = df.Loan_Status
X = df.drop('Loan_Status', axis=1)
```


```python
# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
```

#### Now we do an upsampling to balance out the data. This is necessary because we do not want a model that is biased becaused it learned more of one class than the other. In this case more of 'Yes' in the Loan_Status and less of 'No'


```python
# concatenate our training data back together
X = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)

# separate minority and majority classes
declined = X[X.Loan_Status==0]
approved = X[X.Loan_Status==1]

# upsample minority
declined_upsampled = resample(declined,
                          replace=True, # sample with replacement
                          n_samples=len(approved), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([approved, declined_upsampled])

# check new class counts
upsampled.Loan_Status.value_counts()
```




    1    414
    0    414
    Name: Loan_Status, dtype: int64




```python
#We set our train input and target variables
y_train = upsampled.Loan_Status
X_train = upsampled.drop('Loan_Status', axis=1)
```

#### Now we are ready to fit our data into classifiers. For this project, we use two (2) algorithms; K - NN algorithm and Random Forest algorithm. We compare their performnces and we chooose whichever one performs better.


```python
# Fitting K-NN to the Training set
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 3)
classifier.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=3, p=3,
                         weights='uniform')




```python
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred
```




    array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1,
           1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
           1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,
           1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int64)




```python
# Measuring Accuracy
print('The accuracy of KNN is: ', metrics.accuracy_score(y_pred, y_test))
```

    The accuracy of KNN is:  0.8187919463087249



```python
# Making confusion matrix
print(confusion_matrix(y_test, y_pred))
```

    [[  3  11]
     [ 16 119]]



```python
classifier = RandomForestClassifier(n_estimators = 3, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='entropy', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=3,
                           n_jobs=None, oob_score=False, random_state=0, verbose=0,
                           warm_start=False)




```python
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred
```




    array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,
           1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
           1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,
           1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=int64)




```python
# Measuring Accuracy
print('The accuracy of RandomForest is: ', metrics.accuracy_score(y_pred, y_test))
```

    The accuracy of RandomForest is:  0.9194630872483222



```python
# Making confusion matrix
print(confusion_matrix(y_test, y_pred))
```

    [[  9   5]
     [  7 128]]



```python
# Now we should save our selected model so that we can easily re-use it at a later time
filename = 'RaFor_model.pkl'
joblib.dump(classifier, filename)
```




    ['RaFor_model.pkl']



#### Now that we saved our model, we can try at a later time (this later time could be hours, days or even months) with new data and expect it to perform well (as it has been trained to do). Here we use a csv file (test.csv) of shape 9 x 11. Notice that test.csv does not have the Loan_Status column but only has the input variable.


```python
trial = pd.read_csv(r'C:\Users\STEPHEN EDWIN\test.csv')
```


```python
trial.head()
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
      <th>Gender</th>
      <th>Married</th>
      <th>Guarantor</th>
      <th>Guarantor_Contribution</th>
      <th>University_Staff</th>
      <th>Income_Level</th>
      <th>LoanAmount</th>
      <th>Credit_History</th>
      <th>Union_member</th>
      <th>pledge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>75000</td>
      <td>15000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>100000</td>
      <td>45000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>55000</td>
      <td>14000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>50000</td>
      <td>30000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>50000</td>
      <td>15000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = joblib.load(r'C:\Users\STEPHEN EDWIN\RaFor_model.pkl')
y_pred = model.predict(trial)
new_pred = list(map(lambda x: 'Approved' if x == 1 else 'Declined', y_pred))
val = pd.DataFrame(new_pred, columns=['Status'])
val
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
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Declined</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Approved</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Declined</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Approved</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Declined</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Approved</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Approved</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Approved</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Declined</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Approved</td>
    </tr>
  </tbody>
</table>
</div>


### Conclusion
### We have been able to successfully build a predictive model for loan approval. We were able to save our trained model. We also re-used our model at a later time and saw that it performed as expected.
