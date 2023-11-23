import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
def strIntoAscii(column, dataframe):
   dataframe[column] = dataframe[column].astype(str)
   for j in range(len(dataframe)):
    diff = 0
    for char in  dataframe.at[j, column]:
        diff = diff + ord(char)
    dataframe.at[j, column] = str(diff)
   dataframe[column] = dataframe[column].astype("int64")
   

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# step 1 
df.drop(labels=["Name", "PassengerId"], axis=1, inplace=True)

# fixxing Sex
for i in range(len(df)):
    if df.at[i, "Sex"] == "male":
        df.at[i, "Sex"] = "0"
    else:
        df.at[i,"Sex"] = "1"

df["Sex"] = df["Sex"].astype("int32")

strIntoAscii("Cabin",df)
strIntoAscii("Embarked", df)
# fix Cabin

# fix ticket
df["Ticket"] = df["Ticket"].str.extract("(\d+)", expand=False)

df["Ticket"] = df["Ticket"].astype("float")
df = df.dropna()

# labels = classes that we want to predict
labels =  df.iloc[:, [0]]

# attributes are the attributes that we want to change
attributes = df.loc[:, df.columns != "Survived"]


XTrain, Xtest , yTrain, yTest = train_test_split(df.drop("Survived", axis=1), df["Survived"])

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

