import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the preprocessed dataset
data = pd.read_csv('collision_risk_dataset_preprocessed.csv')

# Split features and target
X = data.drop(columns=['collision_risk'])
y = data['collision_risk']

# Perform a stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Initialize the XGBoost Classifier with adjusted hyperparameters
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=3,
    scale_pos_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Cross-validation
cv = StratifiedKFold(n_splits=5)
cross_val_scores = cross_val_score(xgb_model, X, y, cv=cv)

print("Cross-validation Accuracy:", cross_val_scores.mean())

# Fit the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'\n✅ Model Accuracy: {accuracy:.4f}')
print('\nConfusion Matrix:\n', conf_matrix)
print('\nClassification Report:\n', class_report)

# Save the updated model
import joblib
joblib.dump(xgb_model, 'updated_collision_risk_model.pkl')

print("✅ Updated XGBoost model saved successfully!")
