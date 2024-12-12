from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import pandas as pd

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "your_secret_key"

db = SQLAlchemy(app)

# Define the User model (with hashed password)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

class PredictionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    input_data = db.Column(db.Text)
    risk_level = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# Load the pre-trained model
# model = pickle.load(open('heart_disease_rf_model.pkl', 'rb'))
with open('model.obj', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')  # Correct hash method

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please log in.')
            return redirect(url_for('login'))

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('input_data'))
        flash('Invalid email or password.')
    return render_template('login.html')

@app.route('/input', methods=['GET', 'POST'])
def input_data():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_id = session['user_id']

        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        thalach = int(request.form['thalach'])
        
        # Collecting only the 6 required features from the form
        user_input = {
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'thalach': [thalach]
        }
        user_DF = pd.DataFrame(user_input)

        # Ensure input data is in the correct format (2D array with only 6 features)
        # prediction = model.predict([list(input_data.values())])[0]  # Wrap the input data in a list to make it 2D
        
        # prediction = model.predict(user_DF)

        # print(model['classifier'])

        prediction = model['classifier'].predict(user_DF)
        
        # Determine the risk level based on the prediction
        risk_level, risk_message = determine_risk_level(prediction, user_DF)

        # Log the prediction to the database
        log = PredictionLog(user_id=user_id, input_data=str(input_data), risk_level=risk_level)
        db.session.add(log)
        db.session.commit()

        return render_template('result.html', risk_level=risk_level, risk_message=risk_message)
    
    return render_template('input.html')

# def determine_risk_level(prediction):
#     # Define the risk levels based on the prediction score
#     if prediction < 60:
#         return "Level 1: No heart disease. You're healthy. Maintain a balanced diet and exercise regularly."
#     elif prediction < 70:
#         return "Level 2: Low risk. However, please consult a doctor in a nearby clinic."
#     elif prediction < 80:
#         return "Level 3: Heart disease risk detected. Please immediately consult a doctor and follow dietary recommendations."
#     else:
#         return "Level 4: Emergency admission required! Medication needed."

def determine_risk_level(prediction, user_DF):

    cp_value = user_DF['cp'].values[0]
    chol_value = user_DF['chol'].values[0]
    thalach_value = user_DF['thalach'].values[0]

    if cp_value > 2 and chol_value > 240 and thalach_value > 180:
        return "Level 4: - Urgency", "Emergency admission to the hospital needed! Medication required."
    # Level 3: Heart disease risk detected
    elif cp_value > 1 and chol_value > 200 and thalach_value > 150:
        return "Level 3: - High Risk", "Heart disease risk detected. Please immediately consult a doctor and follow dietary recommendations from your doctor."

    # Level 2: Low risk, consult a doctor
    if prediction == 0:
        return "Level 2: - Low Risk", "Low risk detected. However, please consult a doctor in a nearby clinic."

    # Level 1: Healthy, no heart disease
    return "Level 1: - Healthy", "No heart disease detected. You're healthy. Maintain a balanced diet and exercise regularly"


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():  # Ensure the application context is pushed
        db.create_all()  # Create tables
    app.run(debug=True)