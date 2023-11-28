import pandas as pd
import numpy as np
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

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
X=df.drop(y.columns,axis = 1)
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Finally, initialize a LogisticRegression object with a liblinear solver, and fit it to the training data.
logisticRegr = LogisticRegression(solver='liblinear')
logisticRegr.fit(X_train, y_train)
print(y_train.isnull().sum())
logisticRegr.predict(X_test[0:10])
predictions = logisticRegr.predict(X_test)

cls = ClassifierMixin()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from scikit-learn.
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')