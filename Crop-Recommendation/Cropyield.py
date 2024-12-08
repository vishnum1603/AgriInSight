import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, concatenate
from tensorflow.keras.optimizers import Adam 

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

def create_hybrid_model(input_shape_mlp, input_shape_lstm):
    mlp_input = Input(shape=(input_shape_mlp,))
    mlp_output = Dense(64, activation='relu')(mlp_input)
    mlp_output = Dropout(0.2)(mlp_output)

    lstm_input = Input(shape=(input_shape_lstm[0], input_shape_lstm[1]))
    lstm_output = LSTM(50, activation='relu', return_sequences=True)(lstm_input)
    lstm_output = LSTM(50, activation='relu')(lstm_output)
    lstm_output = Dropout(0.2)(lstm_output)

    combined = concatenate([mlp_output, lstm_output])

    output = Dense(1)(combined)

    model = Model(inputs=[mlp_input, lstm_input], outputs=output)

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

def recommend_crop(data, state, district, season):
    filtered_data = data[(data['State_Name'] == state) & 
                         (data['District_Name'] == district) & 
                         (data['Season'] == season)]

    crop_group = filtered_data.groupby('Crop')['Production'].mean().reset_index()

    recommended_crop = crop_group.loc[crop_group['Production'].idxmax()]['Crop']
    
    return int(recommended_crop)  

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

    filtered_data = preprocess_data(filtered_data)[0]

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(filtered_data[['Area', 'Production']])

    X_mlp = scaled_features[:, 0].reshape(-1, 1)  
    X_lstm = scaled_features[:, 0].reshape(-1, 1, 1)  

    input_shape_mlp = X_mlp.shape[1]
    input_shape_lstm = (X_lstm.shape[1], X_lstm.shape[2])

    model = create_hybrid_model(input_shape_mlp, input_shape_lstm)
    model.fit([X_mlp, X_lstm], scaled_features[:, 1], epochs=0, batch_size=32, verbose=1)

    input_mlp = np.array([[area]])
    input_lstm = np.array([[[area]]])

    predicted_yield_scaled = model.predict([input_mlp, input_lstm])
    predicted_yield = scaler.inverse_transform([[0, predicted_yield_scaled[0][0]]])[0][1]
    
    return predicted_yield

data = pd.read_csv('dataset/Agriculture In India.csv') 

data, label_encoder_state, label_encoder_district, label_encoder_season, label_encoder_crop = preprocess_data(data)

state = input("Enter the state: ").strip().upper() 
district = input("Enter the district: ").strip().upper()
season = input("Enter the season: ").strip().upper()
area = float(input("Enter the area in hectares: "))

state_encoded = label_encoder_state.transform([state])[0]
district_encoded = label_encoder_district.transform([district])[0]
season_encoded = label_encoder_season.transform([season])[0]

recommended_crop_encoded = recommend_crop(data, state_encoded, district_encoded, season_encoded)

recommended_crop = label_encoder_crop.inverse_transform([recommended_crop_encoded])[0]
print(f"Recommended crop for {state}, {district} in {season} season is: {recommended_crop}")

predicted_yield = predict_yield(data, state, district, season, recommended_crop, area, (label_encoder_state, label_encoder_district, label_encoder_season, label_encoder_crop))
print(f"Predicted Yield for {recommended_crop} in {area} hectares: {predicted_yield:.2f} tons")
