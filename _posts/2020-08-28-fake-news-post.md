---
title: "Fake News Detection Project"
date: 2020-08-28
tags: [data wrangling, data science, fake news detection]
header:
  image: ""
excerpt: "Data Wrangling, Data Science, Fake News Detection"
mathjax: "true"
---
# Fake news detection


```python
#Lets import the necessary libraries
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```

## The Dataset
### The dataset used for this project (news.csv) has a shape of 7796×4 and is about 29.2MB. The first column contains the news, the second contains the title, the third cotains the text while the fourth column contains the labels which denote whether the news is REAL or FAKE. You can download it [here](https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view)


```python
#Read the data
df=pd.read_csv(r'C:\Users\STEPHEN EDWIN\Documents\news.csv')
#Get shape and head
df.shape
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
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8476</td>
      <td>You Can Smell Hillary’s Fear</td>
      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10294</td>
      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>
      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3608</td>
      <td>Kerry to go to Paris in gesture of sympathy</td>
      <td>U.S. Secretary of State John F. Kerry said Mon...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10142</td>
      <td>Bernie supporters on Twitter erupt in anger ag...</td>
      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>875</td>
      <td>The Battle of New York: Why This Primary Matters</td>
      <td>It's primary day in New York and front-runners...</td>
      <td>REAL</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Now lets get the labels
labels=df.label
labels.head()
```




    0    FAKE
    1    FAKE
    2    REAL
    3    FAKE
    4    REAL
    Name: label, dtype: object




```python
#Now we split the dataset
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
```

### Now we are going to use the TF-IDF vectorizer.
#### TF stands for Term Frequency which is the number of times a word appears in a document. A higher value means a term appears more often than others, and so, the document is a good match when the term is part of the search terms.
#### IDF stands for Inverse Document Frequency which refers to how common or rare a word is in the entire document set. The closer it is to 0, the more common a word is.
#### You can read more about TF_IDF vectorizer [here](https://monkeylearn.com/blog/what-is-tf-idf/)


```python
#Now we initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

#Next we fit and transform the train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)
```

### What are Passive Aggressive algorithms?
#### Passive Aggressive algorithms are online learning algorithms. They remain passive for a correct classification outcome but they turn aggressive when there's a miscalculation, updating and adjusting. They do not converge since Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.


```python
#Now we need to initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Now we go on to predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
```

    Accuracy: 92.9%



```python
#Now lets build a confusion matrix to see how our performed
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
```




    array([[591,  47],
           [ 43, 586]], dtype=int64)



### So with this model, we have 591 true positives, 586 true negatives, 43 false positives, and 47 false negatives, with an accuracy of 92.9% ###
