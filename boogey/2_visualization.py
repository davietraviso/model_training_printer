import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import pandas as pd
import numpy as np

# Load predictions
predictions = pd.read_excel("2_predictions.xlsx")

# Muat model dan data
model = joblib.load('2_random_forest_model.pkl')
X_train = pd.read_csv('2_X_train.csv')

# Scatter Plot (Actual vs Predicted)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=predictions["Actual"], y=predictions["Predicted"], alpha=0.6)
plt.plot([0, max(predictions["Actual"])], [0, max(predictions["Actual"])], color='red', linestyle='--')
plt.title("Scatter Plot: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# Residual Plot
residuals = predictions["Actual"] - predictions["Predicted"]
plt.figure(figsize=(8, 6))
sns.scatterplot(x=predictions["Predicted"], y=residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual Plot: Predicted vs Residual")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.show()

# Histogram of Residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=20)
plt.axvline(0, color='red', linestyle='--')
plt.title("Histogram of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

# Feature Importance (if RandomForest model is available)
importances = model.feature_importances_
feature_names = X_train.columns  # Replace with actual feature names
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
