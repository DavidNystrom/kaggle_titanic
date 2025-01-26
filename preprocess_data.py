import pandas as pd
from sklearn.model_selection import train_test_split

# Load datasets
train_data = pd.read_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/train.csv')
test_data = pd.read_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/test.csv')

# Combine datasets for consistent preprocessing (excluding the target variable in test data)
test_data['Survived'] = None
combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

# Handle missing values
# Fill missing 'Age' with median
combined_data['Age'] = combined_data['Age'].fillna(combined_data['Age'].median())

# Fill missing 'Embarked' with the most common value
combined_data['Embarked'] = combined_data['Embarked'].fillna(combined_data['Embarked'].mode()[0])

# Fill missing 'Fare' with median
combined_data['Fare'] = combined_data['Fare'].fillna(combined_data['Fare'].median())

# For 'Cabin', create a new feature indicating whether the cabin is known or not
combined_data['CabinKnown'] = combined_data['Cabin'].notnull().astype(int)

# Drop the 'Cabin' and 'Ticket' columns as they are less informative
combined_data.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

# Feature engineering
# Create a new feature for family size
combined_data['FamilySize'] = combined_data['SibSp'] + combined_data['Parch'] + 1

# Map 'Sex' to numeric values
combined_data['Sex'] = combined_data['Sex'].map({'male': 0, 'female': 1})

# One-hot encode the 'Embarked' feature
combined_data = pd.get_dummies(combined_data, columns=['Embarked'], drop_first=True)

# Split the combined data back into train and test sets
train_processed = combined_data[combined_data['Survived'].notnull()]
test_processed = combined_data[combined_data['Survived'].isnull()]

# Separate features and target variable for the training set
X = train_processed.drop(['Survived', 'Name', 'PassengerId'], axis=1)
y = train_processed['Survived']

# Drop unnecessary columns in the test set
X_test = test_processed.drop(['Survived', 'Name', 'PassengerId'], axis=1)

# Train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data for modeling
X_train.to_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/X_train.csv', index=False)
X_val.to_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/X_val.csv', index=False)
y_train.to_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/y_train.csv', index=False)
y_val.to_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/y_val.csv', index=False)
X_test.to_csv('/Users/davidnystrom/Desktop/School/2025/Projects/titanic/data/X_test.csv', index=False)

print("Preprocessing complete. Data saved for modeling.")
