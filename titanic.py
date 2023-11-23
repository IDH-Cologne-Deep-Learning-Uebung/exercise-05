import array
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df.drop('Name', axis=1, inplace=True)
print(df)
df.drop('PassengerId', axis=1, inplace=True)
print(df)
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
print(df) # male = 0; female = 1
df['Cabin'] = pd.to_numeric(df['Cabin'].str.extract('(\d+)', expand=False))
print(df)
df['Ticket'] = pd.to_numeric(df['Ticket'].str.extract('(\d+)', expand=False))
print(df)
df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
print(df) # Aus S wird 0, aus C wird 1, aus Q wird 2

# 3. Remove all rows that contain missing values
df.dropna(subset=['Age'], inplace=True)
print(df)
df.dropna(subset=['Cabin'], inplace=True)
print(df)


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
y=df.iloc[:,[-1]]
x=df.drop(y.columns,axis = 1)
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
sklearn.model_selection.train_test_split()

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.

