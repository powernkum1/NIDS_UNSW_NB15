import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load datasets
train_df = pd.read_csv("data/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("data/UNSW_NB15_testing-set.csv")

# Binary classification target
y_train = train_df["label"]
y_test = test_df["label"]

# Drop unnecessary columns
X_train = train_df.drop(["label", "attack_cat"], axis=1)
X_test = test_df.drop(["label", "attack_cat"], axis=1)

# Encode categorical columns
combined = pd.concat([X_train, X_test])

for col in combined.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# Split back
X_train = combined.iloc[:len(X_train)]
X_test = combined.iloc[len(X_train):]

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Naive Bayes
# -----------------------------
print("\\nNaive Bayes Results")

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

nb_predictions = nb_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))
print(confusion_matrix(y_test, nb_predictions))

# -----------------------------
# Random Forest
# -----------------------------
print("\\nRandom Forest Results")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))
print(confusion_matrix(y_test, rf_predictions))