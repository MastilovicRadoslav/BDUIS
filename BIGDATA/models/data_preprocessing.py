import pandas as pd

def load_and_preprocess_data(filepath):
    """
    Učitava i priprema podatke za treniranje modela.
    Uklanja nedostajuće vrednosti i razdvaja ulazne (X) i izlazne (y) podatke.
    """
    # Učitavanje podataka
    data = pd.read_csv(filepath)
    print("Originalne dimenzije skupa podataka:", data.shape)

    # Pozivanje pripreme podataka za dalju obradu
    data = prepare_input_data(data)

    # Uklanjanje nedostajućih vrednosti
    data = data.dropna()
    print("Dimenzije posle uklanjanja nedostajućih vrednosti:", data.shape)

    # Razdvajanje na ulazne (X) i izlazne (y) podatke
    X = data[['AirTemperature', 'CloudOpacity', 'DHI', 'DNI', 'EBH', 'GHI']]
    y = {
        'Production - Location 1': data['Production - Location 1'],
        'Production - Location 2': data['Production - Location 2'],
        'Production - Location 3': data['Production - Location 3']
    }
    return X, y

def group_data_by_interval(filepath, interval='daily'):
    """
    Grupisanje podataka po dnevnim ili mesečnim intervalima.
    """
    data = pd.read_csv(filepath)
    
    # Eksplicitna konverzija u datetime sa odgovarajućim formatom
    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d-%m-%y %H:%M', errors='coerce')

    if interval == 'daily':
        grouped_data = data.groupby(data['Datetime'].dt.date).mean()
    elif interval == 'monthly':
        grouped_data = data.groupby(data['Datetime'].dt.to_period('M')).mean()
    else:
        raise ValueError("Interval može biti 'daily' ili 'monthly'.")
    
    X = grouped_data[['AirTemperature', 'CloudOpacity', 'DHI', 'DNI', 'EBH', 'GHI']]
    y = {
        'Production - Location 1': grouped_data['Production - Location 1'],
        'Production - Location 2': grouped_data['Production - Location 2'],
        'Production - Location 3': grouped_data['Production - Location 3']
    }
    return X, y


def prepare_input_data(data):
    """
    Pretvara kolone u numeričke vrednosti i uklanja redove sa nedostajućim podacima.
    """
    for column in ['AirTemperature', 'CloudOpacity', 'DHI', 'DNI', 'EBH', 'GHI']:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna()
    return data
