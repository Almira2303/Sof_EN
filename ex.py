from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model
model_rf = joblib.load('heart_disease.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    user_input = {feature: [float(request.form[feature])] for feature in features}
    user_DF = pd.DataFrame(user_input)

    # Make a prediction using the trained model
    pred_user = model_rf.predict(user_DF)[0]

    # Extracting input values for decision making
    cp_value = user_DF['cp'].values[0]
    chol_value = user_DF['chol'].values[0]
    thalach_value = user_DF['thalach'].values[0]
    thal_value = user_DF['thal'].values[0]
    restecg_value = user_DF['restecg'].values[0]

    # Determine urgency level
    if cp_value > 2 and chol_value > 240 and thalach_value > 180:
        level = 5
    elif cp_value > 1 and chol_value > 200 and thalach_value > 150 and thal_value > 2 and restecg_value > 1:
        level = 3
    else:
        level = None

    # Render the result page with prediction and urgency level
    return render_template('result.html', level=level, pred_user=pred_user, cp_value=cp_value, chol_value=chol_value)

if __name__ == '__main__':
    app.run(debug=True)