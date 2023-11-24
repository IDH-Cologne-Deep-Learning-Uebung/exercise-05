import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
   

# read the data from a CSV file (included in the repository)
titanic = pd.read_csv("data/train.csv")


# when we hot encode it, then there is no need to drop this labels
# if u still wanna do it uncomment this line
# df = df.drop(labels=["Name", "PassengerId"], axis=1, inplace=True)

titanic = pd.get_dummies(titanic).dropna()

# step 1 
XTrain, Xtest , yTrain, yTest = train_test_split(titanic.drop("Survived", axis=1), titanic["Survived"])

LogReg = LogisticRegression(solver="liblinear")

LogReg.fit(XTrain,yTrain)
predictedValues = LogReg.predict(Xtest)
ModelScore = LogReg.score(Xtest, yTest)
presision = precision_score(yTest, predictedValues)
recall = recall_score(yTest, predictedValues)
print( "score: " + str( ModelScore))
print("presision: " + str(presision))
print("recall: " + str(recall))

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
# 3. Remove all rows that contain missing values


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.

