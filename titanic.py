import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# Step 3: Data Preprocessing
# 1. Remove the columns "Name" and "PassengerId"
df = df.drop(["Name", "PassengerId"], axis=1)

# 2. Convert non-numeric columns into numeric ones
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Cabin"] = pd.factorize(df["Cabin"])[0]
df["Ticket"] = pd.factorize(df["Ticket"])[0]
df["Embarked"] = pd.factorize(df["Embarked"])[0]

# 3. Remove rows with missing values
df = df.dropna()

# Step 4: Split data and Train Logistic Regression Model
# 1. Split input features (X) and labels (y)
X = df.drop("Survived", axis=1)
y = df["Survived"]

# 2. Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize Logistic Regression model
logreg_model = LogisticRegression(solver='liblinear')

# 4. Fit the model to the training data
logreg_model.fit(X_train, y_train)

# 5. Predict on the test data
y_pred = logreg_model.predict(X_test)

# 6. Calculate precision, recall, and f1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
