import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# read the data from a CSV file (included in the repository)
df = pd.read_csv("exercise-05/data/train.csv")


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
# 3. Remove all rows that contain missing values

#testing        #use survived and drowned   
#print(df.head())
#print(df.tail(3))
#print(df.columns)
#print(df.index)
#print(df["Name"])

#1remove:  name, passangerId
del df["Name"]
del df["PassengerId"]
print(df.columns)
print()

#2 make numeric = "Sex", "Cabin", "Ticket" and "Embarked"
df["Sex"] = pd.factorize(df["Sex"], use_na_sentinel=False)[0]
#print(df["Sex"])
df["Cabin"] = pd.factorize(df["Cabin"], use_na_sentinel=False)[0]       #richtig? -wonach?
print(df["Cabin"])
df["Ticket"] = pd.to_numeric(df["Ticket"], errors="coerce")             #richtig? -wonach?
print(df["Ticket"])
df["Embarked"] = pd.factorize(df["Embarked"], use_na_sentinel=False)[0]
#df["Embarked"] = pd.to_numeric(df["Embarked"], errors="coerce")
#print(df["Embarked"])

print()

#df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce")
#df[["Sex", "Cabin", "Ticket", "Embarked"]] = df[["Sex", "Cabin", "Ticket", "Embarked"]].apply(pd.to_numeric)
#not really perfect -> cabin, ticket = types?
#mgl gleiches in 1 Line...

#3 remove all !rows! with: missing values
#df.dropna(axis="index", how="all")

#df.dropna(axis="index")
dfN = df.dropna(axis="index")
#print(df.dropna(axis="index"))
#print("dfNow: ",df)

#print(dfN)

#print(df.tail(4))
#worked????


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.



#1 split input features from training labels    -pandas
#correct known/want: survived=1, (drowned)=0  <- ["Survived"]
    #training labels = output = y
#y = np.array[[0][0]]
#print("Sur", df.loc[1,"Survived"])

y = dfN["Survived"].to_numpy  #1D-Array falsch...   ->need 2d...
#y = dfN.to_numpy
#y = np.array(y)

#y().reshape(521, 1)
#y().reshape(-1,1)
#print("y: ", y)
#print(y)
    #input features = x
del dfN["Survived"]
X = dfN.to_numpy 
#X = np.array(X)
#X().reshape(-1,1)
print("dfN: ")
print(dfN.tail(3))
print("X:")
#print(X())

#print(y().shape)
#print(X().shape)

#->was gemeint?

#2 split train and test data    -sklearn.model_selection.train_test_split()
#np = df.to_numpy
#X, y = np.arange(10).reshape((5, 2)), range(5)
#print("x: ",X)
#print("y:",y)

print("output?: ",train_test_split(X, y, test_size=0.1, random_state=42))
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#3
# create object
#initilize object       -with liblinear solver
cls = LogisticRegression(solver="liblinear")  #classifier
#train  fit()
cls.fit(x_train, y_train)

#4

#predictions on test set   predict() 
#x_train, y_train, x_test, y_test
y_pred = cls.predict(x_test)
print(y_pred)
#evaluate
try:
    #print()
    score = cls.score(y_test, y_pred)            #*_score(y_test, y_pred)
    print(score)
    #print(cls.score(y_test, y_pred))
except ValueError:
    #print("error")
    raise

#: lengths = [_num_samples(X) for X in arrays if X is not None]
# line 103  - train_test_split
#error: TypeError: Singleton array array() cannot be considered a valid collection.