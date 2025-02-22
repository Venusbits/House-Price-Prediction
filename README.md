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

Installation & Setup

Clone the repository:

git clone https://github.com/Venusbits/house-price-prediction.git
cd house-price-prediction

Install dependencies:

pip install -r requirements.txt

Run the Flask app:

python app.py

Open a browser and visit:

http://127.0.0.1:5000/

Usage

Enter the house details such as area, number of bedrooms, bathrooms, stories, etc.

Click on the "Predict" button.

The estimated price will be displayed.

Dataset Details

The dataset used is Housing.csv, which contains house features and their corresponding prices. The dataset includes the following attributes:

Area

Bedrooms

Bathrooms

Stories

Main Road Access

Air Conditioning

Parking

Preferred Area

Price (Target Variable)

Model Training

The dataset is preprocessed in model.py

Categorical variables are encoded

The dataset is split into training and testing sets

A Linear Regression model is trained and evaluated

The trained model is saved as model.pkl

Future Improvements

Enhance model accuracy by using advanced machine learning models (e.g., Random Forest, XGBoost)

Improve UI with better frontend styling

Deploy the app on Heroku or Render

Contributing

Feel free to fork the repository and make improvements. Contributions are welcome!
