import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import mlflow

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import joblib
import json
from typing import List

# Define the preprocessor outside the function if it's not going to change
#preprocessor = joblib.load('preprocessor.joblib')
#model = joblib.load('linear_regression_model.joblib')

# Helper function to create DataFrame
def preprocessing (model_key: str,
                            mileage: float,
                           engine_power: float,
                           fuel: str,
                           paint_color: str,
                           car_type: str,
                           private_parking_available: bool,
                           has_gps: bool,
                           has_air_conditioning: bool,
                           automatic_car: bool,
                           has_getaround_connect: bool,
                           has_speed_regulator: bool,
                           winter_tires: bool): #-> pd.DataFrame:

    data = {
        'model_key': [model_key],
        'mileage': [mileage],
        'engine_power': [engine_power],
        'fuel': [fuel],
        'paint_color': [paint_color],
        'car_type': [car_type],
        'private_parking_available': [private_parking_available],
        'has_gps': [has_gps],
        'has_air_conditioning': [has_air_conditioning],
        'automatic_car': [automatic_car],
        'has_getaround_connect': [has_getaround_connect],
        'has_speed_regulator': [has_speed_regulator],
        'winter_tires': [winter_tires]
    }
    df = pd.DataFrame(data)
    print(df)

    boolean_columns = [
    'private_parking_available',
    'has_gps',
    'has_air_conditioning',
    'automatic_car',
    'has_getaround_connect',
    'has_speed_regulator',
    'winter_tires'
]
# Convert 'Yes' to 1 and 'No' to 0
    df[boolean_columns] = df[boolean_columns].replace({True: 1, False: 0})
    print(df)

    categorical_features = ['model_key','fuel','paint_color','car_type']

    #numerical_features = df.columns.drop(categorical_features)
    numerical_features = ['mileage', 'engine_power'] + boolean_columns

    #print (numerical_features)
    numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

    categorical_transformer = Pipeline(
    steps=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore')) # first column will be dropped to avoid creating correlations between features
    ])
    #print('print1')

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    #print(2)

    preprocessor = joblib.load('preprocessor.joblib')
    #print(3)
    # Load the model from the file
    model = joblib.load('linear_regression_model.joblib')
  
    x = preprocessor.transform(df)
    print(x)

    # Make predictions
    prediction = model.predict(x)  # Convert to list for JSON serializability
    prediction_rounded = [round(p, 2) for p in prediction]
    # Return JSON response
    result = ({"prediction": prediction_rounded})

    return result



### 
# Here you can define some configurations 
###
description = """
## Provide us with the following in formation and we will be able to predict the price of your car rental:


* brand
* mileage
* engine power
* type of fuel
* color
* type of car
* availability of private parking
* gps
* air conditioning 
* automatic car
* connect check-in
* speed regulator
* winter tires

## API Endpoint Documentation for "Get Around: Predict the Price of Your Car Rental"

* Endpoint: /predict_price

* HTTP Method: POST

* Description: This endpoint takes car details as input and returns the predicted rental price of the car.
    Required Input:
    
        * brand (string): The brand of the car (e.g., "Toyota").
        * mileage (int): The mileage of the car in kilometers (e.g., 35000).
        * engine_power (int): The power of the car's engine in horsepower (e.g., 110).
        * type_of_fuel (string): The type of fuel the car uses (e.g., "diesel").
        * color (string): The color of the car (e.g., "black").
        * type_of_car (string): The category of the car (e.g., "SUV").
        * private_parking_available (bool): Whether private parking is available (e.g., "yes" or "no").
        * gps (bool): Whether the car has a GPS (e.g., "yes" or "no").
        * air_conditioning (bool): Whether the car has air conditioning (e.g., "yes" or "no").
        * automatic_car (bool): Whether the car is automatic (e.g., "yes" or "no").
        * connect_check_in (bool): Whether the car supports GetAround Connect for remote check-in (e.g., "yes" or "no").
        * speed_regulator (bool): Whether the car has a speed regulator (e.g., "yes" or "no").
        * winter_tires (bool): Whether the car has winter tires (e.g., "yes" or "no").

* Expected Output:

    - Response: JSON object with the predicted price of the car rental.
        Example: {"predicted_price": 49.99}





"""
#tag_metadata = [
#    {
#        "name": "Name_1",
#        "description": "LOREM IPSUM NEC."
#    },

#    {
#        "name": "Name_2",
#        "description": "LOREM IPSUM NEC."
#    }
#]

app = FastAPI(
    title = 'Get Around: predict the price of your car rental',
    description = description, 
    #openapi_tags=tag_metadata
)


#class PredictionFeatures(BaseModel):
#    YearsExperience: float    

###
# Here you define enpoints 
###
@app.post("/predict", tags=["PredictPrice"])
async def PredictPrice(
                       model_key: str, 
                       mileage: float,
                       engine_power: float,
                       fuel: str,
                       paint_color: str,
                       car_type: str,
                       private_parking_available: bool,
                       has_gps: bool,
                       has_air_conditioning: bool,
                       automatic_car: bool,
                       has_getaround_connect: bool,
                       has_speed_regulator: bool,
                       winter_tires: bool):
    

    result = preprocessing(model_key, mileage, engine_power, fuel, paint_color, car_type,
                                     private_parking_available, has_gps, has_air_conditioning,
                                     automatic_car, has_getaround_connect, has_speed_regulator,
                                     winter_tires)
    
    return {"prediction": result}
    """
    Process a new text from Elon Musk and return a prediction of the evolution of the Tesla Quote.
    """
    #result = preprocessing(text, date, ouverture)


#@app.post("/predict", tags=["Machine Learning"])
#async def predict(predictionFeatures: PredictionFeatures):
    """
    Prediction of salary for a given year of experience! 
    """
    # Read data 
    #years_experience = pd.DataFrame({"YearsExperience": [predictionFeatures.YearsExperience]})

    # Log model from mlflow 
    #logged_model = 'runs:/5e54b2ee620546b0914c9e9fbfd18875/salary_estimator'

    # Load model as a PyFuncModel.
    #loaded_model = mlflow.pyfunc.load_model(logged_model)
    #prediction = loaded_model.predict(years_experience)
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4005)
