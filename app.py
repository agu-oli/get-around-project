import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ValidationError
from fastapi.encoders import jsonable_encoder


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Define the CarDetails model
class CarDetails(BaseModel):
    brand: str
    mileage: float = Field(..., gt=0, description="Mileage must be greater than zero.")
    engine_power: float = Field(..., gt=0, description="Engine power must be greater than zero.")
    fuel: str = Field(..., description="Fuel type of the car.")
    paint_color: str = Field(..., description="Paint color of the car.")
    car_type: str = Field(..., description="Type of car.")
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

    @field_validator('brand')
    def model_must_be_valid(cls, v):
        valid_brand = ['citroën', 'peugeot', 'pgo', 'renault', 'audi', 'bmw', 'ford', 'mercedes', 'opel', 'porsche', 'volkswagen', 'kia motors', 'alfa romeo', 'ferrari', 'fiat', 'lamborghini', 'maserati', 'lexus', 'honda', 'mazda', 'mini', 'mitsubishi', 'nissan', 'seat', 'subaru', 'suzuki', 'toyota', 'yamaha']
        if v.lower() not in valid_brand:
            raise ValueError(f"Brand '{v}' is not valid. Choose from {valid_brand}")
        return v.lower()

    @field_validator('fuel')
    def fuel_must_be_valid(cls, v):
        valid_fuel_types = ['diesel', 'petrol', 'hybrid petrol', 'electro']
        if v.lower() not in valid_fuel_types:
            raise ValueError(f"Fuel type '{v}' is not valid. Choose from {valid_fuel_types}")
        return v.lower()

    @field_validator('paint_color')
    def paint_color_must_be_valid(cls, v):
        valid_colors = ['black', 'grey', 'white', 'red', 'silver', 'blue', 'orange', 'beige', 'brown', 'green']
        if v.lower() not in valid_colors:
            raise ValueError(f"Paint color '{v}' is not valid. Choose from {valid_colors}")
        return v.lower()

    @field_validator('car_type')
    def car_type_must_be_valid(cls, v):
        valid_car_types = ['convertible', 'coupe', 'estate', 'hatchback', 'sedan', 'subcompact', 'suv', 'van']
        if v.lower() not in valid_car_types:
            raise ValueError(f"Car type '{v}' is not valid. Choose from {valid_car_types}")
        return v.lower()



# Define the input model for the prediction
class CarInput(BaseModel):
    input: list[list]  # Expecting a list of lists

class PredictionResponse(BaseModel):
    prediction: list[float]    


preprocessor = joblib.load('preprocessor.joblib')
model = joblib.load('linear_regression_model.joblib')

# Define the preprocessor function
def preprocessing(car_details: CarDetails):
    data = {
        'brand': [car_details.brand],
        'mileage': [car_details.mileage],
        'engine_power': [car_details.engine_power],
        'fuel': [car_details.fuel],
        'paint_color': [car_details.paint_color],
        'car_type': [car_details.car_type],
        'private_parking_available': [car_details.private_parking_available],
        'has_gps': [car_details.has_gps],
        'has_air_conditioning': [car_details.has_air_conditioning],
        'automatic_car': [car_details.automatic_car],
        'has_getaround_connect': [car_details.has_getaround_connect],
        'has_speed_regulator': [car_details.has_speed_regulator],
        'winter_tires': [car_details.winter_tires]
    }

    df = pd.DataFrame(data)

    # Transform boolean columns
    boolean_columns = [
        'private_parking_available',
        'has_gps',
        'has_air_conditioning',
        'automatic_car',
        'has_getaround_connect',
        'has_speed_regulator',
        'winter_tires'
    ]

    df[boolean_columns] = df[boolean_columns].replace({True: 1, False: 0})

    categorical_features = ['brand', 'fuel', 'paint_color', 'car_type']
    numerical_features = ['mileage', 'engine_power'] + boolean_columns


    # Preprocess the data
    x = preprocessor.transform(df)

    # Make predictions
    prediction = model.predict(x)
    prediction_rounded = [round(p, 2) for p in prediction]

    return {"prediction": prediction_rounded}


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

* Endpoints: 
            /predict_price  (accessible via python and you terminal)
            /predict_from_browser (insert the data and predict the price on your browser)   

* HTTP Method: POST

* Description: This endpoint takes car details as input and returns the predicted rental price of the car.
    - Required Input:
    
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

    - Errors in the request will be treated with the corresponding correction in the respose body.

* Expected Output:

    - Response: JSON object with the predicted price of the car rental.
        Example: {"predicted_price": 49.99}


"""


# FastAPI app initialization
app = FastAPI(title='Get Around: predict the price of your car rental',
             description = description)


# the corresponding value error message will be shown if the input variables is not correct

# exception handler for Pydantic ValidationError
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValueError):
    error_message = exc.errors()[0]['msg']
    return JSONResponse(
        status_code=400,
        content={"detail": error_message},
    )

    #return JSONResponse(
    #    status_code=422,
    #    content=jsonable_encoder({"detail": exc.errors()}),
    #)


@app.post("/predict_price")
async def PredictPrice(car_input: CarInput):
    input = car_input.input  # Extract the list from the CarInput object

    # Extract details from the input list
    car_details = CarDetails(
        brand=input[0][0],
        mileage=input[0][1],
        engine_power=input[0][2],
        fuel=input[0][3],
        paint_color=input[0][4],
        car_type=input[0][5],
        private_parking_available=bool(input[0][6]),
        has_gps=bool(input[0][7]),
        has_air_conditioning=bool(input[0][8]),
        automatic_car=bool(input[0][9]),
        has_getaround_connect=bool(input[0][10]),
        has_speed_regulator=bool(input[0][11]),
        winter_tires=bool(input[0][12]),
    )

    result = preprocessing(car_details)
    return result




@app.post("/predict_from_browser", response_model=PredictionResponse, tags=["PredictPrice"])
async def PredictPrice(
    brand: str,
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
    winter_tires: bool,
):
    try:   
    # Create a CarDetails object from the redeclared parameters
        car_details = CarDetails(
            brand=brand,
            mileage=mileage,
            engine_power=engine_power,
            fuel=fuel,
            paint_color=paint_color,
            car_type=car_type,
            private_parking_available=private_parking_available,
            has_gps=has_gps,
            has_air_conditioning=has_air_conditioning,
            automatic_car=automatic_car,
            has_getaround_connect=has_getaround_connect,
            has_speed_regulator=has_speed_regulator,
            winter_tires=winter_tires
        )

        result = preprocessing(car_details)
        return result
    except ValueError as e:
        raise e  # This will be caught by the custom value error handler

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4005)