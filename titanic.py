import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")
print(df)


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df.pop('Name')
df.pop('PassengerId')


print(" 'Name' und 'PassengerId' entfernt:")
print(df.head()) #head um nur die ersten Ergebnisse zu sehen
#print(df)
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".

df[['Sex', 'Cabin', 'Ticket', 'Embarked']] = df[['Sex', "Cabin", "Ticket", "Embarked"]].apply(pd.to_numeric, errors = 'ignore')

print("Nach der Konvertierung der Spalten:")
print(df.head())

df_new=df.dropna() #entfernt alle Zeilen, die leere Zellen enthalten
print(df_new)

# ## Step 4

# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`. TODO
# Trennen von Eingabemerkmalen (Features) und Zielvariable (Label)
X = df_new.drop('Survived', axis=1)  # Hier werden alle Spalten außer 'Survived' als Eingabemerkmale ausgewählt
Y = df_new['Survived']  # die Spalte 'Survived' als Zielvariable ausgewählt

# Anzeigen der ersten paar Zeilen der Eingabemerkmale und der Zielvariable
print("Eingabemerkmale (Features):")
print(X.head())

print("\nZielvariable (Survived):")
print(y.head())


# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Ausgabe der Formen der Trainings- und Testdaten
print("Form der Trainingsdaten (x_train, y_train):", x_train.shape, y_train.shape)
print("Form der Testdaten (x_test, y_test):", x_test.shape, y_test.shape)

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
logreg_model = LogisticRegression(solver='liblinear')
logreg_model.fit(x_train, y_train)


# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.

y_pred = logreg_model.predict(x_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f_score = f1_score(y_test, y_pred)

# Ausgaben Precision,Recall und f-Score
print("Precision:", precision)
print("Recall:", recall)
print("F-Score:", f_score)