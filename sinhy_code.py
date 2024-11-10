# Import necessary libraries
from shiny import App, render, ui, reactive
import pandas as pd
import pickle  # Assuming the trained model is saved as a pickle file
from sklearn.preprocessing import MinMaxScaler  # Assuming MinMaxScaler was used

# Load your pre-trained model
with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler used for feature transformation
with open('MinMaxScaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the UI of the Shiny application
app_ui = ui.page_fluid(
    ui.h2("House Price Prediction"),
    ui.input_text("zip_code", "Enter ZIP Code:", value=""),
    ui.input_numeric("bedrooms", "Number of Bedrooms:", value=3),
    ui.input_numeric("bathrooms", "Number of Bathrooms:", value=2),
    ui.input_numeric("year_built", "Year Built:", value=2000),
    ui.input_select("home_type", "Home Type:", 
                    choices=["homeType_CONDO", "homeType_LOT", "homeType_MANUFACTURED", "homeType_MULTI_FAMILY", "homeType_SINGLE_FAMILY", "homeType_TOWNHOUSE"],
                    selected="homeType_SINGLE_FAMILY"),  # Add your list of home types here
    ui.input_numeric("lot_size", "Lot Size (in sq ft):", value=5000),
    ui.output_text("prediction_output")
)

# Define server logic
def server(input, output, session):

    # Reactive function to handle input and make predictions
    @reactive.Calc
    def predict_price():
        # Collect input data from UI
        zip_code = input.zip_code()
        bedrooms = input.bedrooms()
        bathrooms = input.bathrooms()
        year_built = input.year_built()
        home_type = input.home_type()  # Dropdown value for home_type
        lot_size = input.lot_size()
        
        # Create DataFrame with input data
        user_input_features = pd.DataFrame({
            'zipcode': [zip_code],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'year_built': [year_built],
            'home_type': [home_type],
            'lot_size': [lot_size]
        })
        
        # Apply scaling to input features
        user_input_features_scaled = scaler.transform(user_input_features)
        
        # Predict the scaled price
        predicted_price_scaled = model.predict(user_input_features_scaled)[0]
        
        # Unscale the predicted price
        predicted_price = scaler.inverse_transform([[predicted_price_scaled]])[0][0]

        # Format and return the unscaled prediction as currency
        return f"${predicted_price:,.2f}"
    
    # Render the prediction output
    output.prediction_output = render.text(predict_price)

# Create and run the app
app = App(app_ui, server)
