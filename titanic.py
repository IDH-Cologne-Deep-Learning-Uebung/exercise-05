
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df = df.drop(['Name', 'PassengerId'], axis=1) 
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
df['Sex', 'Cabin', 'Ticket'] = pd.to_numeric(df['Sex', 'Cabin', 'Ticket'], errors='coerce')
# 3. Remove all rows that contain missing values
df = df.dropna()


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
x = df.iloc[:, :-1]
y = df.iloc[:, -1] 
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
logR = LogisticRegression(solver='liblinear')
logR.fit(x_train, y_train)
y_pred = logR.predict(x_test)
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f_score = f1_score(y_test, y_pred)

print('Precision:', precision)
print('Recall:', recall)
print('F-Score:', f_score)
