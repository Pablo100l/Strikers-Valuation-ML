import tensorflow as tf
import numpy as np
import joblib

# 1. Wczytanie zapisanego modelu oraz skalerów
model = tf.keras.models.load_model("model_napastnikow.keras")
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")

print("=== SYSTEM WYCENY NAPASTNIKÓW (AI) ===")
print("Wprowadź statystyki zawodnika:")

# 2. Pobieranie danych od użytkownika przez inputy
mecze = float(input("Liczba meczów: "))
gole = float(input("Liczba goli: "))
asysty = float(input("Liczba asyst: "))
wiek = float(input("Wiek zawodnika: "))
rok = float(input("Rok: "))

print("\nWybierz ligę:")
print("1. Premier League")
print("2. La Liga")
print("3. Serie A")
print("4. Bundesliga")
print("5. Ligue 1")
wybor_ligę = input("Twój wybór (1-5): ")

# Inicjalizacja zmiennych One-Hot dla lig
pl, lal, sa, bun, lig = 0, 0, 0, 0, 0

if wybor_ligę == '1': pl = 1
elif wybor_ligę == '2': lal = 1
elif wybor_ligę == '3': sa = 1
elif wybor_ligę == '4': bun = 1
elif wybor_ligę == '5': lig = 1
else:
    print("Nieznany wybór ligi, domyślnie ustawiono brak (wszystkie 0).")

# 3. Tworzenie wektora danych
nowy_zawodnik = np.array([[mecze, gole, asysty, wiek, rok, pl, lal, sa, bun, lig]])

# 4. Skalowanie i predykcja
nowy_zawodnik_scaled = scaler_x.transform(nowy_zawodnik)
predykcja_scaled = model.predict(nowy_zawodnik_scaled, verbose=0)
predykcja_cena = scaler_y.inverse_transform(predykcja_scaled)

# 5. Wynik
print(f"\nPrzewidywana wartość rynkowa zawodnika wynosi: {predykcja_cena[0][0]:.2f} mln €")