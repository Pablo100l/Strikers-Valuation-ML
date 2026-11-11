import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# 1. Ładowanie danych i ponowne odtworzenie podziału (żeby mieć dokładnie te same 80 elementów)
dane = pd.read_excel("zawodnicy.xlsx", sheet_name="Arkusz1")
dane = pd.get_dummies(dane, columns=['liga'])

X = dane[['mecze', 'gole', 'asysty', 'wiek', 'rok', 
          'liga_Premier League', 'liga_La Liga', 'liga_Serie A', 'liga_Bundesliga', 'liga_Ligue 1']].values
y = dane['wartosc'].values
zawodnicy_array = dane['zawodnik'].values

x_train_raw, x_test_raw, y_train_raw, y_test_raw, imiona_train, imiona_test = train_test_split(
    X, y, zawodnicy_array, test_size=0.2, random_state=42
)

# 2. Ładowanie modelu i skalerów z dysku
model = tf.keras.models.load_model("model_napastnikow.keras")
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# 3. Skalowanie i predykcja dla zbioru treningowego
x_train_scaled = scaler_x.transform(x_train_raw)
predykcje_skalowane = model.predict(x_train_scaled)
predykcje = scaler_y.inverse_transform(predykcje_skalowane).flatten()

# 4. Rysowanie wykresu
x_indeksy = np.arange(len(y_train_raw))
szerokosc = 0.35

plt.figure(figsize=(14, 6))
plt.bar(x_indeksy - szerokosc/2, y_train_raw, szerokosc, label='Rzeczywista wartość', color='blue')
plt.bar(x_indeksy + szerokosc/2, predykcje, szerokosc, label='Przewidywania modelu', color='orange')

plt.xlabel('Zawodnik (Zbiór Treningowy)')
plt.ylabel('Wartość (mln €)')
plt.title('Porównanie cen: Rzeczywistość vs Model (Zbiór Treningowy)')
plt.xticks(x_indeksy, imiona_train, rotation=65, fontsize=8)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()