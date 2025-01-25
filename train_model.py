from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2)  # Split into train/test

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Print accuracy (line length fixed)
print("Accuracy: {}".format(accuracy))  # Now within the 79-character limit

# Save the model
joblib.dump(model, 'model.pkl')  # Removed trailing space