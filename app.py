import pandas as pd
from flask import Flask, render_template, request
import numpy as np

# Machine Learning Imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# -------------------------------------------------------------------
# 1. LOAD AND CLEAN DATA
# -------------------------------------------------------------------
try:
    # Load the CSV data
    car = pd.read_csv('cleaned_data.csv')
except FileNotFoundError:
    print("CRITICAL ERROR: 'cleaned_data.csv' not found. App will crash.")
    car = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type', 'Price'])

if not car.empty:
    # Clean hidden spaces and format data
    car['company'] = car['company'].astype(str).str.strip()
    car['name'] = car['name'].astype(str).str.strip()
    car['year'] = car['year'].astype(int).astype(str)

    # Ensure consistent naming for 'Other' category
    def clean_model_name(row):
        company = row['company']
        model = row['name']
        if company in model:
            final_name = model
        else:
            final_name = company + " " + model

        if company == 'other' and 'other' not in final_name:
            final_name = 'other ' + final_name
        return final_name

    car['name'] = car.apply(clean_model_name, axis=1)

# -------------------------------------------------------------------
# 2. CREATE A NEW MODEL INSTANTLY
# -------------------------------------------------------------------
pipe = None

if not car.empty:
    X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = car['Price']

    # Setup the transformer (Handle Categories)
    ohe = OneHotEncoder()
    # We fit the encoder on the data
    ohe.fit(X[['name', 'company', 'year', 'fuel_type']])

    column_trans = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'year', 'fuel_type']),
        remainder='passthrough'
    )

    # Setup Linear Regression
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)

    # Train the model
    pipe.fit(X, y)
    print("------------------------------------------------")
    print("SUCCESS: Model trained on-the-fly and ready!")
    print("------------------------------------------------")


# -------------------------------------------------------------------
# 3. WEB INTERFACE
# -------------------------------------------------------------------
@app.route('/')
def index():
    # Fill the dropdowns
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    return render_template('index.html',
                           companies=companies,
                           car_models=car_models,
                           years=years,
                           fuel_types=fuel_types)


@app.route('/predict', methods=['POST'])
def predict():
    if pipe is None:
        return "Error: Model could not be trained (CSV missing?)"

    # 1. Get data from user
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # 2. Safety Check: Convert kilometers to number
    try:
        driven = int(driven)
    except:
        driven = 0  # Default if user left it blank

    # 3. Prepare data for model
    # The columns MUST be in this exact order
    input_data = pd.DataFrame([[car_model, company, year, driven, fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # 4. Predict
    try:
        prediction = pipe.predict(input_data)

        # Round to 2 decimal places
        price = np.round(prediction[0], 2)

        # --- NEW LOGIC: Check for negative price ---
        if price < 0:
            return "0(Car is too old to sell)"
        else:
            return str(price)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)