House Price Prediction

This project is a House Price Prediction Web App built using Flask and a machine learning model trained with Scikit-learn. It allows users to input various features of a house and predict its estimated price.

Features

Predicts house prices based on input features

Uses a Linear Regression model trained on housing data

Implements a Flask web application for easy user interaction

Encodes categorical variables and scales numerical features for better prediction accuracy

Technologies Used

Python (Flask, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

Flask (for web framework)

Joblib (for model persistence)

HTML & CSS (for frontend)

Project Structure

├── app.py           # Flask application

├── model.py         # Model training script

├── model.pkl        # Trained model

├── templates/

│   ├── index.html   # Frontend HTML template

├── Housing.csv      # Dataset 

├── cleaned_housing.csv  # Preprocessed dataset

├── README.md        # Project documentation
