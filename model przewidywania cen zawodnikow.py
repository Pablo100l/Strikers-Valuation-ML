import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


dane = pd.read_excel("zawodnicy.xlsx", sheet_name="Arkusz1")
dane = pd.get_dummies(dane, columns=['liga'])


x_train = dane[['mecze', 'gole', 'asysty', 'wiek', 'rok',
                         'liga_Premier League', 'liga_La Liga', 'liga_Serie A', 'liga_Bundesliga', 'liga_Ligue 1']].values
y_train = dane['wartosc'].values


zawodnicy_array = dane['zawodnik'].values
x_train_raw, x_test_raw, y_train_raw, y_test_raw, imiona_train, imiona_test = train_test_split(
    x_train, y_train, zawodnicy_array, test_size=0.2, random_state=42
)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train_raw)
x_test_scaled = scaler_x.transform(x_test_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)),  
    tf.keras.layers.Dense(units=40, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="linear")
])
 

model.compile(optimizer='adam',
              loss='mean_absolute_percentage_error',
              metrics=['mae', 'mse'])

model.fit(x_train_scaled, y_train_scaled, epochs=300)

blad = model.evaluate(x_test_scaled, y_test_scaled)[0]
print(f"Błąd modelu (Loss) na zbiorze testowym: {blad:.4f}")

model.save("model_napastnikow.keras")
joblib.dump(scaler_x, "scaler_x.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")