import pandas as pd

"""
  Učitava i priprema podatke za treniranje modela.
- Uklanja nedostajuće vrednosti.
- Razdvaja ulazne (X) i izlazne (y) podatke.
"""
def load_and_preprocess_data(filepath):
    # Učitavanje podataka iz CSV fajla
    data = pd.read_csv(filepath)
    print("Originalne dimenzije skupa podataka:", data.shape)  # Prikaz originalnih dimenzija

    # Priprema podataka (konverzija u numeričke vrednosti i uklanjanje nedostajućih podataka)
    data = prepare_input_data(data)

    # Uklanjanje redova sa nedostajućim vrednostima
    data = data.dropna()
    print("Dimenzije posle uklanjanja nedostajućih vrednosti:", data.shape)  # Prikaz dimenzija nakon čišćenja

    # Razdvajanje ulaznih podataka (X) i ciljeva (y)
    X = data[['AirTemperature', 'CloudOpacity', 'DHI', 'DNI', 'EBH', 'GHI']]  # Ulazni parametri
    y = {
        'Production - Location 1': data['Production - Location 1'],  # Ciljna promenljiva za Lokaciju 1
        'Production - Location 2': data['Production - Location 2'],  # Ciljna promenljiva za Lokaciju 2
        'Production - Location 3': data['Production - Location 3']   # Ciljna promenljiva za Lokaciju 3
    }
    return X, y  # Vraća ulazne podatke i ciljeve

def prepare_input_data(data):
    """
    Pretvara kolone u numeričke vrednosti i uklanja redove sa nedostajućim podacima.
    """
    # Konverzija podataka u numeričke vrednosti
    for column in ['AirTemperature', 'CloudOpacity', 'DHI', 'DNI', 'EBH', 'GHI']:
        data[column] = pd.to_numeric(data[column], errors='coerce')  # Pretvara podatke u float, uklanja nevalidne vrednosti
    
    # Uklanjanje redova sa nedostajućim vrednostima
    data = data.dropna()
    return data  # Vraća očišćene podatke
