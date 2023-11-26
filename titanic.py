import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/train.csv")

df.pop('Name')
df.pop('PassengerId')

non_numeric_columns = ['Ticket']
for column in non_numeric_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

df = pd.DataFrame(df)
df['Cabin'] = df['Cabin'].fillna('')

def extract_numeric(x):
    match = re.search(r'\d+', str(x))
    return int(match.group(0)) if match else np.nan

df['Cabin'] = df['Cabin'].apply(extract_numeric)
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'Q': 1, 'C':2})

df.dropna(inplace=True)

#set labels manualy from processed cvs
df = pd.read_csv('output_file.csv', header=0,
                 names=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch','Ticket','Fare','Cabin','Embarked'])

labels = df.iloc[:, 0]  # Assuming labels are in the first column
data = df.iloc[:, 1:]

#print(data)
#print(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)

y_pred = logistic_regression_model.predict(X_test)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.4f}")#precision goes down for every fale positive
print(f"Recall: {recall:.4f}")#higher mean fewer fale negatives
print(f"F1 Score: {f1:.4f}") # f1 is both



# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
# 3. Remove all rows that contain missing values


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.

