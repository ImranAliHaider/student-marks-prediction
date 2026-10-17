import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the illustrative dataset
data = pd.read_csv("student_marks.csv")
X = data[["study_hours"]]
y = data["marks"]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model and predict the test rows
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Results on illustrative sample data")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, predictions):.2f} marks")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, predictions)):.2f} marks")
print(f"R-squared (R2): {r2_score(y_test, predictions):.3f}")
print("\\nActual vs predicted:")
for actual, predicted in zip(y_test.tolist(), predictions.tolist()):
    print(f"Actual: {actual:>5.1f} | Predicted: {predicted:>5.1f}")
