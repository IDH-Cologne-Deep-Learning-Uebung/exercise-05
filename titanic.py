from distutils.log import Log
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).

df = df.drop(columns=["Name", "PassengerId"])

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
df["Sex"]=pd.factorize(df.Sex)[0]
df["Cabin"]=pd.factorize(df.Cabin)[0]
df["Ticket"]=pd.factorize(df.Ticket)[0]
df["Embarked"]=pd.factorize(df.Embarked)[0]

# 3. Remove all rows that contain missing values
df = df.dropna()


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
x = df.drop(columns =["Survived"])
y = df.Survived

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.33, random_state=42)

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
model = LogisticRegression(solver='liblinear', random_state=0).fit(x_train, y_train)

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
