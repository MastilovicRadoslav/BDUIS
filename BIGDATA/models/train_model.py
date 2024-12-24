from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.data_preprocessing import load_and_preprocess_data, group_data_by_interval
import pandas as pd

def create_model_pipeline():
    """
    Kreira pipeline sa standardizacijom i regresorom.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

# Funkcija za evaluaciju modela
def evaluate_model(model, X_test, y_test, label):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"{label}: MAE={mae:.2f}, MSE={mse:.2f}, R²={r2:.2f}")
    return mae, mse, r2

# Funkcija za evaluaciju ukupne proizvodnje (sabiranje predikcija)
def evaluate_total_production(models, X_test, y_test_total, label):
    pred_1 = models[0].predict(X_test)
    pred_2 = models[1].predict(X_test)
    pred_3 = models[2].predict(X_test)
    total_pred_sum = pred_1 + pred_2 + pred_3
    mae = mean_absolute_error(y_test_total, total_pred_sum)
    mse = mean_squared_error(y_test_total, total_pred_sum)
    r2 = r2_score(y_test_total, total_pred_sum)
    print(f"{label}: MAE={mae:.2f}, MSE={mse:.2f}, R²={r2:.2f}")
    return mae, mse, r2

# SATNI MODELI
X, y = load_and_preprocess_data('dataset/Data_Cacak.csv')
X_train, X_test, y_train, y_test = train_test_split(X, pd.DataFrame(y), test_size=0.2, random_state=42)

model_location_1 = create_model_pipeline()
model_location_2 = create_model_pipeline()
model_location_3 = create_model_pipeline()

model_location_1.fit(X_train, y_train['Production - Location 1'])
model_location_2.fit(X_train, y_train['Production - Location 2'])
model_location_3.fit(X_train, y_train['Production - Location 3'])

print("\n--- SATNI MODELI ---")
evaluate_model(model_location_1, X_test, y_test['Production - Location 1'], "Satni model - Lokacija 1")
evaluate_model(model_location_2, X_test, y_test['Production - Location 2'], "Satni model - Lokacija 2")
evaluate_model(model_location_3, X_test, y_test['Production - Location 3'], "Satni model - Lokacija 3")

# Evaluacija za ukupnu proizvodnju
evaluate_total_production(
    [model_location_1, model_location_2, model_location_3], 
    X_test, y_test.sum(axis=1), "Satna ukupna proizvodnja"
)

# DNEVNI MODELI
X_daily, y_daily = group_data_by_interval('dataset/Data_Cacak.csv', 'daily')
y_daily_df = pd.DataFrame(y_daily)
X_daily_train, X_daily_test, y_daily_train, y_daily_test = train_test_split(X_daily, y_daily_df, test_size=0.2, random_state=42)

model_daily_1 = create_model_pipeline()
model_daily_2 = create_model_pipeline()
model_daily_3 = create_model_pipeline()

model_daily_1.fit(X_daily_train, y_daily_train['Production - Location 1'])
model_daily_2.fit(X_daily_train, y_daily_train['Production - Location 2'])
model_daily_3.fit(X_daily_train, y_daily_train['Production - Location 3'])

print("\n--- DNEVNI MODELI ---")
evaluate_model(model_daily_1, X_daily_test, y_daily_test['Production - Location 1'], "Dnevni model - Lokacija 1")
evaluate_model(model_daily_2, X_daily_test, y_daily_test['Production - Location 2'], "Dnevni model - Lokacija 2")
evaluate_model(model_daily_3, X_daily_test, y_daily_test['Production - Location 3'], "Dnevni model - Lokacija 3")

# Evaluacija za ukupnu proizvodnju
evaluate_total_production(
    [model_daily_1, model_daily_2, model_daily_3], 
    X_daily_test, y_daily_test.sum(axis=1), "Dnevna ukupna proizvodnja"
)

# MESEČNI MODELI
X_monthly, y_monthly = group_data_by_interval('dataset/Data_Cacak.csv', 'monthly')
y_monthly_df = pd.DataFrame(y_monthly)
X_monthly_train, X_monthly_test, y_monthly_train, y_monthly_test = train_test_split(X_monthly, y_monthly_df, test_size=0.2, random_state=42)

model_monthly_1 = create_model_pipeline()
model_monthly_2 = create_model_pipeline()
model_monthly_3 = create_model_pipeline()

model_monthly_1.fit(X_monthly_train, y_monthly_train['Production - Location 1'])
model_monthly_2.fit(X_monthly_train, y_monthly_train['Production - Location 2'])
model_monthly_3.fit(X_monthly_train, y_monthly_train['Production - Location 3'])

print("\n--- MESEČNI MODELI ---")
evaluate_model(model_monthly_1, X_monthly_test, y_monthly_test['Production - Location 1'], "Mesečni model - Lokacija 1")
evaluate_model(model_monthly_2, X_monthly_test, y_monthly_test['Production - Location 2'], "Mesečni model - Lokacija 2")
evaluate_model(model_monthly_3, X_monthly_test, y_monthly_test['Production - Location 3'], "Mesečni model - Lokacija 3")

# Evaluacija za ukupnu proizvodnju
evaluate_total_production(
    [model_monthly_1, model_monthly_2, model_monthly_3], 
    X_monthly_test, y_monthly_test.sum(axis=1), "Mesečna ukupna proizvodnja"
)
