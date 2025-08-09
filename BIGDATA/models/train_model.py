from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.data_preprocessing import load_and_preprocess_data
import pandas as pd

def create_model_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()), # Normalizuje podatke kako bi imali standardnu distribuciju (srednja vrednost 0 i standardna devijacija 1)
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  # RandomForestRegressor se koristi za predikciju
    ])

# Funkcija za evaluaciju pojedinačnih modela
#MAE (Mean Absolute Error) – kolika je u prosjeku greška
#MSE (Mean Squared Error) – kazna za velike greške
#R² (R-squared) – koliko dobro model objašnjava podatke (bliže 1 = bolje)
def evaluate_model(model, X_test, y_test, label):
    predictions = model.predict(X_test)  # Generisanje predikcija na osnovu test podataka
    mae = mean_absolute_error(y_test, predictions)  # Prosečna apsolutna greška
    mse = mean_squared_error(y_test, predictions)  # Srednja kvadratna greška
    r2 = r2_score(y_test, predictions)  # Koeficijent determinacije (R²)
    print(f"{label}: MAE={mae:.2f}, MSE={mse:.2f}, R²={r2:.2f}")  # Štampanje metrika za evaluaciju modela
    return mae, mse, r2  # Vraćanje vrednosti metrika

# Funkcija za evaluaciju ukupne proizvodnje (sabiranjem predikcija)
def evaluate_total_production(models, X_test, y_test_total, label):
    pred_1 = models[0].predict(X_test)  # Predikcije za Lokaciju 1
    pred_2 = models[1].predict(X_test)  # Predikcije za Lokaciju 2
    pred_3 = models[2].predict(X_test)  # Predikcije za Lokaciju 3
    total_pred_sum = pred_1 + pred_2 + pred_3  # Sabiranje predikcija za sve lokacije
    mae = mean_absolute_error(y_test_total, total_pred_sum)  # Prosečna apsolutna greška za ukupnu proizvodnju
    mse = mean_squared_error(y_test_total, total_pred_sum)  # Srednja kvadratna greška za ukupnu proizvodnju
    r2 = r2_score(y_test_total, total_pred_sum)  # R² za ukupnu proizvodnju
    print(f"{label}: MAE={mae:.2f}, MSE={mse:.2f}, R²={r2:.2f}")  # Štampanje rezultata evaluacije
    return mae, mse, r2

# SATNI MODELI
X, y = load_and_preprocess_data('dataset/Data_Cacak.csv')  # Učitavanje i priprema podataka iz CSV fajla
X_train, X_test, y_train, y_test = train_test_split(X, pd.DataFrame(y), test_size=0.2, random_state=42)  # Podela podataka na trening i test skupove

# Kreiranje i treniranje modela za svaku lokaciju, skicit-learn
model_location_1 = create_model_pipeline()
model_location_2 = create_model_pipeline()
model_location_3 = create_model_pipeline()

model_location_1.fit(X_train, y_train['Production - Location 1'])  # Treniranje modela za Lokaciju 1
model_location_2.fit(X_train, y_train['Production - Location 2'])  # Treniranje modela za Lokaciju 2
model_location_3.fit(X_train, y_train['Production - Location 3'])  # Treniranje modela za Lokaciju 3

#Evaulacija
print("\n--- SATNI MODELI ---")
evaluate_model(model_location_1, X_test, y_test['Production - Location 1'], "Satni model - Lokacija 1")  # Evaluacija za Lokaciju 1
evaluate_model(model_location_2, X_test, y_test['Production - Location 2'], "Satni model - Lokacija 2")  # Evaluacija za Lokaciju 2
evaluate_model(model_location_3, X_test, y_test['Production - Location 3'], "Satni model - Lokacija 3")  # Evaluacija za Lokaciju 3

# Evaluacija za ukupnu proizvodnju (sabiranjem)
evaluate_total_production(
    [model_location_1, model_location_2, model_location_3], X_test, y_test.sum(axis=1), "Satna ukupna proizvodnja"
)
