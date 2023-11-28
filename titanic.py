import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df.drop(['PassengerId', 'Name'], axis = 1)
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
df['Cabin'] = df['Cabin'].apply(extract_numeric())
ticketColumns = ['Ticket']
for column in ticketColumns:
    df[column] = pd.to_numeric(df[column], errors='coerce')
df['Embarked'] = df['Embarked'].map({'S': 0, 'Q': 1, 'C':2})
# 3. Remove all rows that contain missing values
df.dropna(how='all')

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
titanic = pd.get_dummies(titanic).dropna()
XTrain, Xtest , yTrain, yTest = train_test_split(titanic.drop("Survived", axis=1), titanic["Survived"])
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
LogReg.fit(XTrain,yTrain)
predictedValues = LogReg.predict(Xtest)
ModelScore = LogReg.score(Xtest, yTest)
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
presision = precision_score(yTest, predictedValues)
recall = recall_score(yTest, predictedValues)

print( "score: " + str( ModelScore))
print("presision: " + str(presision))
print("recall: " + str(recall))

