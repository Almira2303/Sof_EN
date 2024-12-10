import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Import joblib for saving the model

# Importing dataset
x = pd.read_csv(r'train_heart.csv')
y = pd.read_csv(r'test_heart.csv')

# Exploratory Data Analysis
plt.subplots(figsize=(15, 10))
sns.heatmap(x.corr(), annot=True)
plt.show()

# Histogram of cholesterol
sns.histplot(data=x, x='chol')
plt.title('Histogram of Cholesterol')
plt.show()

# Bar plot of chest pain type
sns.countplot(data=x, x='cp', hue='target')
plt.title('Bar Plot of Chest Pain Type')
plt.show()

# Count of males and females having heart disease
sns.countplot(data=x, x='sex', hue='target')
plt.title('Distribution of Heart Disease by Gender')
plt.xlabel('Gender 0 = female, 1 = male')
plt.ylabel('Count')
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()

# Scatter plots
sns.scatterplot(data=x, x='age', y='chol', hue='target')
plt.title('Scatter Plot of Age and Cholesterol with Target')
plt.show()

sns.scatterplot(data=x, x='age', y='trestbps', hue='target')
plt.title('Scatter Plot of Age and Resting Blood Pressure')
plt.show()

# Importing Random Forest Classifier from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Removing target column from data and selecting only 6 features
a_train = x[['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']]  # Only 6 features
b_train = x['target']

a_test = y[['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']]  # Only 6 features
b_test = y['target']

# Training the Random Forest model
model_rf = RandomForestClassifier()
model_rf.fit(a_train, b_train)

# Predictions from Random Forest model
b_pred_rf = model_rf.predict(a_test)

# Evaluating the model
print("RANDOM FOREST CLASSIFIER")
print(classification_report(y_pred=b_pred_rf, y_true=b_test))
print("Accuracy:", accuracy_score(y_true=b_test, y_pred=b_pred_rf))

# Confusion matrix for Random Forest
conf_matrix_rf = confusion_matrix(b_test, b_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Heart Disease', 'Heart Disease'],
            yticklabels=['No Heart Disease', 'Heart Disease'])
plt.title('Confusion Matrix - Random Forest Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Save the model using joblib
joblib.dump(model_rf, 'heart_disease_rf_model.pkl')
print("Model saved as 'heart_disease_rf_model.pkl'")

# Checking model with random user input (only 6 features)
age = int(input("Enter age: "))
sex = int(input("Enter sex (0 = female, 1 = male): "))
cp = int(input("Enter cp (0-3): "))
trestbps = int(input("Enter trestbps: "))
chol = int(input("Enter chol: "))
thalach = int(input("Enter thalach: "))

# Only using the 6 relevant features
user_input = {
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'thalach': [thalach]
}
user_DF = pd.DataFrame(user_input)

# Predictions
def heart_prediction(user_DF):
    pred_user = model_rf.predict(user_DF)
    
    # Extract the relevant values from the user input DataFrame
    cp_value = user_DF['cp'].values[0]
    chol_value = user_DF['chol'].values[0]
    thalach_value = user_DF['thalach'].values[0]

    # Level 4: Emergency admission required
    if cp_value > 2 and chol_value > 240 and thalach_value > 180:
        return "Level 4: Urgency: Emergency admission to the hospital needed! Medication required."

    # Level 3: Heart disease risk detected
    if cp_value > 1 and chol_value > 200 and thalach_value > 150:
        return "Level 3: Heart disease risk detected. Please immediately consult a doctor and follow dietary recommendations from your doctor."

    # Level 2: Low risk, consult a doctor
    if pred_user == 0:
        return "Level 2: Low risk. However, Please consult a doctor in a nearby clinic."

    # Level 1: Healthy, no heart disease
    return "Level 1: No heart disease. You're healthy. Maintain a balanced diet and exercise regularly."

# Display the prediction result
result = heart_prediction(user_DF)
print(result)
