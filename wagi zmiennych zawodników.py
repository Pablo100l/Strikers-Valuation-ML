import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import joblib

# 1. Wczytanie modelu
model = tf.keras.models.load_model("model_napastnikow.keras")

# 2. Pobieranie wag z pierwszej warstwy gęstej (Dense)
wagi = None
for warstwa in model.layers:
    wagi_warstwy = warstwa.get_weights()
    if len(wagi_warstwy) > 0 and wagi_warstwy[0].ndim == 2:
        wagi = wagi_warstwy[0]
        break

if wagi is not None:
    nazwy_cech = [
        'mecze', 'gole', 'asysty', 'wiek', 'rok', 
        'liga_Premier League', 'liga_La Liga', 'liga_Serie A', 'liga_Bundesliga', 'liga_Ligue 1'
    ]
    
    # Obliczamy średnią absolutną wagę dla każdej cechy
    waznosc_cech = np.mean(np.abs(wagi), axis=1)

    print("--- WAŻNOŚĆ POSZCZEGÓLNYCH ZMIENNYCH DLA MODELU ---")
    for cecha, waga in sorted(zip(nazwy_cech, waznosc_cech), key=lambda x: x[1], reverse=True):
        print(f"{cecha}: {waga:.4f}")

    # Wykres poziomych słupków
    plt.figure(figsize=(10, 5))
    plt.barh(nazwy_cech, waznosc_cech, color='purple')
    plt.xlabel('Średnia wartość wag (Ważność cechy)')
    plt.ylabel('Cechy zawodnika')
    plt.title('Co najbardziej wpływa na wycenę zawodnika?')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("Nie udało się odnaleźć wag w podanej strukturze modelu.")