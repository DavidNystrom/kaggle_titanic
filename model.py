import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load preprocessed data
X_train = pd.read_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/X_train.csv')
X_val = pd.read_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/X_val.csv')
y_train = pd.read_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/y_train.csv').squeeze()  # Convert to Series
y_val = pd.read_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/y_val.csv').squeeze()  # Convert to Series

# Train an XGBoost Classifier
model = XGBClassifier(
    n_estimators=10000,         # Number of trees
    learning_rate=0.001,       # Learning rate
    max_depth=4,               # Depth of each tree
    min_child_weight=3,        # Minimum sum of instance weight in a leaf
    subsample=0.8,             # Fraction of data used per tree
    colsample_bytree=0.8,      # Fraction of features used per tree
    reg_alpha=0.1,             # L1 regularization (sparsity)
    reg_lambda=1.0,            # L2 regularization (complexity control)
    scale_pos_weight=1.5,      # Balancing classes (adjust for class imbalance)
    random_state=42
)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on validation data
y_pred = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Save the trained model (optional)
'''
import joblib
joblib.dump(model, '/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/titanic_xgboost_model.pkl')
'''
# Optional: Load and make predictions on test data
X_test = pd.read_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/X_test.csv')
test_predictions = model.predict(X_test)

# Save predictions to a CSV file
output = pd.DataFrame({
    "PassengerId": pd.read_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/test.csv')["PassengerId"],
    "Survived": test_predictions
})
output.to_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/titanic_predictions.csv', index=False)

print("Predictions saved to titanic_predictions.csv")
