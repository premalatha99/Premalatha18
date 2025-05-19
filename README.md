import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Dataset (Using sample diabetes dataset)
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = (diabetes.target > 100).astype(int)  # Convert target to binary for prediction

# 2. Data Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Simple Prediction Interface
def predict_disease(input_data):
    input_df = pd.DataFrame([input_data], columns=diabetes.feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return "High Risk" if prediction[0] == 1 else "Low Risk"

# Example usage
sample_input = X.iloc[0].to_dict()
print("Patient Prediction:", predict_disease(sample_input))
