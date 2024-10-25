# get-around-project


## Car rental price predictor

### The app.py is the script of an API that allows users to predict the price of a car rental.

Users are needed to provide the following features on which the predicting model was trained on:

* brand (string): The brand of the car (e.g., "Toyota").
* mileage (int): The mileage of the car in kilometers (e.g., 35000).
* engine_power (int): The power of the car's engine in horsepower (e.g., 110).
* type_of_fuel (string): The type of fuel the car uses (e.g., "diesel").
* color (string): The color of the car (e.g., "black").
* type_of_car (string): The category of the car (e.g., "suv").
* private_parking_available (bool): Whether private parking is available (e.g., "yes" or "no").
* gps (bool): Whether the car has a GPS (e.g., "yes" or "no").
* air_conditioning (bool): Whether the car has air conditioning (e.g., "yes" or "no").
* automatic_car (bool): Whether the car is automatic (e.g., "yes" or "no").
* connect_check_in (bool): Whether the car supports GetAround Connect for remote check-in (e.g., "yes" or "no").
* speed_regulator (bool): Whether the car has a speed regulator (e.g., "yes" or "no").
* winter_tires (bool): Whether the car has winter tires (e.g., "yes" or "no").

* Errors in the request are treated with the corresponding correction in the respose body.

#### This is the link of the API : https://get-around-app-2b184b2f74b6.herokuapp.com/docs

#### The API /predict_price endpoint can be used on python with following code: 

``
response = requests.post("https://get-around-app-2b184b2f74b6.herokuapp.com/predict_price", json={
    "input": [['Toyota', 140411, 100, 'diesel', 'black', 'convertible', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'yes']]
})
print(response.json())``

Or on terminal (Mac) in the following way:

``curl -i -H "Content-Type: application/json" -X POST -d '{"input": [["Toyota", 140411, 100, "diesel", "black", "convertible", "yes", "yes", "no", "no", "yes", "yes", "yes"]]}' https://get-around-app-2b184b2f74b6.herokuapp.com/predict_price``

#### The API /predict_from_browser endpoint, allows you predict the price of a car rental from your browser. It takes the above-mentioned information as input and returns the predicted rental price of the car.

  -The output is a JSON object with the predicted price.
      Example: {"predicted_price": 49.99}
