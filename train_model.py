import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# 1. LOAD DATASET
data = pd.read_csv("dataset/ckd.csv")

print("Dataset loaded successfully!")
print("Original shape:", data.shape)


# 2. CLEAN COLUMN NAMES
data.columns = data.columns.str.replace("'", "", regex=False)

print("\nColumn names:")
print(data.columns.tolist())


# 3. HANDLE MISSING VALUES
data = data.replace("?", pd.NA)

numeric_columns = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc"
]

categorical_columns = [
    "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad", "appet",
    "pe", "ane"
]

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data[column] = data[column].fillna(data[column].median())

for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])


# 4. CLEAN TARGET COLUMN
data["class"] = (
    data["class"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Keep only valid target values
data = data[data["class"].isin(["ckd", "notckd"])].copy()

print("\nDataset after removing invalid target:")
print(data.shape)


# 5. SEPARATE FEATURES AND TARGET
X = data.drop("class", axis=1)

y = data["class"].map({
    "ckd": 1,
    "notckd": 0
})


# 6. ENCODE CATEGORICAL FEATURES
X = pd.get_dummies(
    X,
    columns=categorical_columns
)

print("\nPreprocessing completed!")
print("X shape:", X.shape)
print("Y shape:", y.shape)

print("\nTarget values:")
print(y.value_counts())


# 7. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData splitting completed!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# 8. TRAIN RANDOM FOREST
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")


# 9. PREDICT
y_pred = model.predict(X_test)


# 10. EVALUATE
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 11. FEATURE IMPORTANCE
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Save feature importance for Streamlit dashboard
feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10).to_string(index=False))

print("\nFeature importance saved successfully!")
print("feature_importance.csv")


# 12. SAVE MODEL
joblib.dump(model, "ckd_model.pkl")

joblib.dump(
    X.columns.tolist(),
    "model_columns.pkl"
)

print("\nModel saved successfully!")
print("ckd_model.pkl")
print("model_columns.pkl")