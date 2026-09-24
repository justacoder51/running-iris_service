from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
X, y = load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Test accuracy: {acc:.4f}")

# Save
joblib.dump(clf, "model.pkl")
print("Saved model.pkl")