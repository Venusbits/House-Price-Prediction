from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')  # Ensure this matches your saved model filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        mainroad = 1 if request.form['mainroad'] == 'yes' else 0
        airconditioning = 1 if request.form['airconditioning'] == 'yes' else 0
        parking = int(request.form['parking'])
        prefarea = 1 if request.form['prefarea'] == 'yes' else 0
        
        # Prepare features for prediction
        features = np.array([[area, bedrooms, bathrooms, stories, mainroad, airconditioning, parking, 
                              prefarea]])

        # Make prediction
        prediction = model.predict(features)[0]  

        return render_template('index.html', prediction=f'Estimated Price: ₹{prediction*(100):,.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
