import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
data = pd.read_csv('collision_risk_dataset_preprocessed.csv')

# Split data into features and target
X = data.drop(columns=['collision_risk'])
y = data['collision_risk']

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# XGBoost model with early stopping and handling imbalance
xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=3,
    scale_pos_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    # Move eval_metric to the XGBClassifier constructor
    eval_metric="logloss"  
)

# Remove eval_metric, early_stopping_rounds, and verbose from fit()
xgb_model.fit(
    X_train, y_train,
    # eval_set=[(X_test, y_test)],  # If needed, use for monitoring during training
    # early_stopping_rounds=10,  # Use early stopping with a callback or other methods
    # verbose=True  # Control output with other methods
)

# ... (Rest of the code remains the same)




# Predict and evaluate
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'\\n✅ Model Accuracy: {accuracy:.4f}')
print('\\nConfusion Matrix:\\n', conf_matrix)
print('\\nClassification Report:\\n', class_report)

# Save the improved model
joblib.dump(xgb_model, 'updated_collision_risk_model.pkl')
print("✅ Updated XGBoost model saved successfully!")


