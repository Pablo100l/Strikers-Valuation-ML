⚽ System Wyceny Klasycznych Napastników (AI / Machine Learning)
Projekt Machine Learning służący do automatycznej wyceny rynkowej klasycznych środkowych napastników (tzw. „dziewiątek”) na podstawie ich statystyk boiskowych oraz ligi, w której występują. Model oparty jest na sieci neuronowej stworzonej w bibliotece TensorFlow/Keras.
Projekt został zaprojektowany w sposób modularny – oddzielono proces trenowania modelu od narzędzi analitycznych i predykcyjnych.

📂 Struktura Projektu
zawodnicy.xlsx – Baza danych zawierająca 100 rekordów klasycznych napastników z różnych sezonów i lig z Top 5.
model przewidywania cen zawodnikow.py – Główny skrypt treningowy; uczy sieć neuronową, dokonuje ewaluacji na zbiorze testowym oraz zapisuje gotowy model i skalery na dysk.
model_napastnikow.keras – Zapisany plik wytrenowanego modelu TensorFlow.
scaler_x.pkl / scaler_y.pkl – Zserializowanane obiekty StandardScaler do prawidłowego skalowania danych wejściowych i wyjściowych.
wycena nowego zawodnika.py – Narzędzie interaktywne pozwalające podać statystyki nowego piłkarza przez konsolę i poznać jego szacowaną wartość w mln €.
wagi zmiennych zawodników.py – Skrypt analizujący i wizualizujący ważność poszczególnych cech (gole, wiek, liga itp.) dla modelu.
wykres napastników dla zbioru treningowego.py – Porównanie cen rzeczywistych z predykcjami modelu na danych treningowych (80%).
wykres napastników dla zbioru testowego.py – Ewaluacja i wizualizacja skuteczności modelu na danych testowych (20% – zawodnicy nieznani modelu).
.gitignore – Konfiguracja plików ignorowanych przez system Git.

🛠️ Wymagania techniczne
Przed uruchomieniem skryptów upewnij się, że masz zainstalowane wymagane biblioteki:

Bash
pip install tensorflow pandas numpy matplotlib scikit-learn joblib openpyxl

🚀 Jak używać projektu?
Aktualizacja bazy lub retrening:
Jeśli dodasz nowych zawodników do pliku zawodnicy.xlsx, uruchom główny plik trenujący:

Bash
python "model przewidywania cen zawodnikow.py"

Wycena nowego piłkarza:
Uruchom skrypt interaktywny i podaj statystyki zawodnika w terminalu:

Bash
python "wycena nowego zawodnika.py"

Analiza modelu i wykresy:
Aby sprawdzić ważność cech: python "wagi zmiennych zawodników.py"
Aby zobaczyć wykresy dopasowania (trening/test): uruchom odpowiednie skrypty wykresów.

📊 Cechy brane pod uwagę przez model
1.Liczba meczów w sezonie
2.Liczba goli
3.Liczba asyst
4.Wiek zawodnika
5.Rok sezonu
6.Liga 
