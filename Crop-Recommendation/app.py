import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
import os
import requests

# Function to preprocess data
def preprocess_data(data):
    data.dropna(inplace=True)

    label_encoder_state = LabelEncoder()
    label_encoder_district = LabelEncoder()
    label_encoder_season = LabelEncoder()
    label_encoder_crop = LabelEncoder()
    
    data['State_Name'] = label_encoder_state.fit_transform(data['State_Name'])
    data['District_Name'] = label_encoder_district.fit_transform(data['District_Name'])
    data['Season'] = label_encoder_season.fit_transform(data['Season'])
    data['Crop'] = label_encoder_crop.fit_transform(data['Crop'])

    return data, label_encoder_state, label_encoder_district, label_encoder_season, label_encoder_crop

# Function to create hybrid model
def create_hybrid_model(input_shape_mlp, input_shape_lstm):
    mlp_input = Input(shape=(input_shape_mlp,))
    mlp_output = Dense(124, activation='relu')(mlp_input)
    mlp_output = Dropout(0.2)(mlp_output)

    lstm_input = Input(shape=(input_shape_lstm[0], input_shape_lstm[1]))
    lstm_output = LSTM(50, activation='relu', return_sequences=True)(lstm_input)
    lstm_output = LSTM(50, activation='relu')(lstm_output)
    lstm_output = Dropout(0.2)(lstm_output)

    combined = concatenate([mlp_output, lstm_output])

    output = Dense(1)(combined)

    model = Model(inputs=[mlp_input, lstm_input], outputs=output)

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

# Function to handle unseen labels
def handle_unseen_label(encoder, value, label_name):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return f"Error: Unseen {label_name} - '{value}'"

# Function to recommend crop
def recommend_crop(data, state, district, season):
    filtered_data = data[(data['State_Name'] == state) & 
                         (data['District_Name'] == district) & 
                         (data['Season'] == season)]

    crop_group = filtered_data.groupby('Crop')['Production'].mean().reset_index()

    recommended_crop = crop_group.loc[crop_group['Production'].idxmax()]['Crop']

    return int(recommended_crop)

# Function to predict yield
def predict_yield(data, state, district, season, crop, area, label_encoders):
    label_encoder_state, label_encoder_district, label_encoder_season, label_encoder_crop = label_encoders

    state_encoded = label_encoder_state.transform([state])[0]
    district_encoded = label_encoder_district.transform([district])[0]
    season_encoded = label_encoder_season.transform([season])[0]
    crop_encoded = label_encoder_crop.transform([crop])[0]

    filtered_data = data[(data['State_Name'] == state_encoded) & 
                         (data['District_Name'] == district_encoded) & 
                         (data['Season'] == season_encoded) & 
                         (data['Crop'] == crop_encoded)].copy()

    if filtered_data.empty:
        return "No data available for the given inputs."

    filtered_data, _, _, _, _ = preprocess_data(filtered_data)

    scaler = MinMaxScaler()
    filtered_data[['Area', 'Production']] = scaler.fit_transform(filtered_data[['Area', 'Production']])
    
    input_mlp = np.array([[area]])
    input_lstm = np.array([[[area]]])

    scaled_input = scaler.transform([[area, 0]])
    input_mlp = scaled_input[:, 0].reshape(-1, 1)
    input_lstm = scaled_input[:, 0].reshape(-1, 1, 1)

    model = create_hybrid_model(input_shape_mlp=input_mlp.shape[1], input_shape_lstm=input_lstm.shape[1:])

    history = model.fit([input_mlp, input_lstm], scaled_input[:, 1], epochs=10, batch_size=32, verbose=1)
    
    predicted_yield_scaled = model.predict([input_mlp, input_lstm])
    predicted_yield = scaler.inverse_transform([[area, predicted_yield_scaled[0][0]]])[0][1]
    predicted_yield = abs(predicted_yield) / 1000

    return predicted_yield, history.history['loss'][-1], history.history['mae'][-1]

# Function to fetch real-time crop price
def fetch_crop_price(crop, state, district):
    api_url = "https://agmarknet.gov.in/pricetrends/"  # Replace with actual API URL
    params = {
        "crop": crop,
        "state": state,
        "district": district
    }
    try:
        response = requests.get(api_url, params=params, timeout=10)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")  # Log the raw content
        response.raise_for_status()  # Raise HTTPError for bad responses
        price_data = response.json()  # Parse JSON
        price = price_data.get('price', None)
        if price:
            return f"₹{price} per quintal"
        else:
            return "Price data unavailable"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except ValueError as e:
        return f"Error parsing JSON: {e}"


# Load data
data = pd.read_csv('dataset/Agriculture In India.csv')

data, label_encoder_state, label_encoder_district, label_encoder_season, label_encoder_crop = preprocess_data(data)

crop_image_mapping = {
    'Wheat': 'crop_images/Wheat.jpg',
    'Rice': 'crop_images/Rice.jpg',
     'Maize': 'crop_images/Maize.jpg',
    'Sugarcane': 'crop_images/Sugarcane.jpg',
    'Coconut' : 'crop_images/Coconut.jpg',
    'Banana' : 'crop_images/Banana.jpg',
    'Ragi' : 'crop_images/Ragi.jpg',
    'Arhur/Tur' : 'crop_images/ArhurTur.jpg',
    'Groundnut' : 'crop_images/Groundnut.jpg',
    'Potato' : 'crop_images/Potato.jpg',
    'Sesamum' : 'crop_images/Sesamum.jpg',
    'Cotton(lint)' : 'crop_images/Cotton(lint).jpg'
    # Add more crops and corresponding image paths here
}

st.title("Smart Farming - Crop Recommendation and Yield Prediction")

st.sidebar.header("Input Parameters")

state = st.sidebar.text_input("Enter State").strip().upper()
district = st.sidebar.text_input("Enter District").strip().upper()
season = st.sidebar.text_input("Enter Season").strip().upper()
area = st.sidebar.number_input("Enter Area (in hectares)", min_value=0.0)

if st.sidebar.button("Recommend Crop and Predict Yield"):
    state_encoded = handle_unseen_label(label_encoder_state, state, 'state')
    district_encoded = handle_unseen_label(label_encoder_district, district, 'district')
    season_encoded = handle_unseen_label(label_encoder_season, season, 'season')

    if isinstance(state_encoded, str) or isinstance(district_encoded, str) or isinstance(season_encoded, str):
        st.error(f"Invalid input: {state_encoded if isinstance(state_encoded, str) else ''} "
                 f"{district_encoded if isinstance(district_encoded, str) else ''} "
                 f"{season_encoded if isinstance(season_encoded, str) else ''}")
    else:
        recommended_crop_encoded = recommend_crop(data, state_encoded, district_encoded, season_encoded)
        recommended_crop = label_encoder_crop.inverse_transform([recommended_crop_encoded])[0]

        predicted_yield, loss, mae = predict_yield(data, state, district, season, recommended_crop, area, 
                                                    (label_encoder_state, label_encoder_district, label_encoder_season, label_encoder_crop))

        crop_image_path = crop_image_mapping.get(recommended_crop, None)
        if crop_image_path and os.path.exists(crop_image_path):
            st.image(crop_image_path, caption=f"{recommended_crop}", use_column_width=True)
        else:
            st.warning(f"No image available for {recommended_crop}.")
            
        st.success(f"Recommended Crop: {recommended_crop}")
        st.success(f"Predicted Yield for given area {area}: {predicted_yield * area:.2f} tons")
        
        crop_price = fetch_crop_price(recommended_crop, state, district)
        st.info(f"Real-Time Price for {recommended_crop}: {crop_price}")
