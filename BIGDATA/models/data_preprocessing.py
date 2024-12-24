import pandas as pd

def load_and_preprocess_data(filepath):
    """
    Učitava i priprema podatke za treniranje modela.
    Uklanja nedostajuće vrednosti i razdvaja ulazne (X) i izlazne (y) podatke.
    Dodata provera grešaka u ulaznim podacima.
    """
    try:
        # Učitavanje podataka
        data = pd.read_csv(filepath)
        print("Originalne dimenzije skupa podataka:", data.shape)

        # Provera da li postoje nedostajuće vrednosti
        if data.isnull().values.any():
            print("Upozorenje: Skup podataka sadrži nedostajuće vrednosti!")
        else:
            print("Podaci nemaju nedostajuće vrednosti.")

        # Pozivanje pripreme podataka za dalju obradu
        data = prepare_input_data(data)

        # Uklanjanje nedostajućih vrednosti
        data = data.dropna()
        print("Dimenzije posle uklanjanja nedostajućih vrednosti:", data.shape)

        # Provera formata kolona
        required_columns = ['AirTemperature', 'CloudOpacity', 'DHI', 'DNI', 'EBH', 'GHI', 
                            'Production - Location 1', 'Production - Location 2', 'Production - Location 3']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Kolona '{column}' nedostaje u skupu podataka.")

        # Razdvajanje na ulazne (X) i izlazne (y) podatke
        X = data[['AirTemperature', 'CloudOpacity', 'DHI', 'DNI', 'EBH', 'GHI']]
        y = {
            'Production - Location 1': data['Production - Location 1'],
            'Production - Location 2': data['Production - Location 2'],
            'Production - Location 3': data['Production - Location 3']
        }
        return X, y
    except Exception as e:
        print(f"Greška prilikom učitavanja i pripreme podataka: {e}")
        raise

def group_data_by_interval(filepath, interval='daily'):
    """
    Grupisanje podataka po dnevnim ili mesečnim intervalima.
    Dodata provera grešaka u ulaznim podacima i validacija intervala.
    """
    try:
        data = pd.read_csv(filepath)
        print("Originalne dimenzije skupa podataka:", data.shape)

        # Provera validnosti kolone 'Datetime'
        if 'Datetime' not in data.columns:
            raise ValueError("Kolona 'Datetime' nedostaje u skupu podataka.")

        # Eksplicitna konverzija u datetime sa odgovarajućim formatom
        data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d-%m-%y %H:%M', errors='coerce')

        if data['Datetime'].isnull().any():
            raise ValueError("Neuspešna konverzija 'Datetime' kolone u odgovarajući format.")

        # Grupisanje po zadatom intervalu
        if interval == 'daily':
            grouped_data = data.groupby(data['Datetime'].dt.date).mean()
        elif interval == 'monthly':
            grouped_data = data.groupby(data['Datetime'].dt.to_period('M')).mean()
        else:
            raise ValueError("Interval može biti samo 'daily' ili 'monthly'.")

        # Provera da li postoje potrebne kolone
        required_columns = ['AirTemperature', 'CloudOpacity', 'DHI', 'DNI', 'EBH', 'GHI', 
                            'Production - Location 1', 'Production - Location 2', 'Production - Location 3']
        for column in required_columns:
            if column not in grouped_data.columns:
                raise ValueError(f"Kolona '{column}' nedostaje u grupisanim podacima.")

        X = grouped_data[['AirTemperature', 'CloudOpacity', 'DHI', 'DNI', 'EBH', 'GHI']]
        y = {
            'Production - Location 1': grouped_data['Production - Location 1'],
            'Production - Location 2': grouped_data['Production - Location 2'],
            'Production - Location 3': grouped_data['Production - Location 3']
        }
        return X, y
    except Exception as e:
        print(f"Greška prilikom grupisanja podataka: {e}")
        raise


def prepare_input_data(data):
    """
    Pretvara kolone u numeričke vrednosti i uklanja redove sa nedostajućim podacima.
    """
    try:
        for column in ['AirTemperature', 'CloudOpacity', 'DHI', 'DNI', 'EBH', 'GHI']:
            data[column] = pd.to_numeric(data[column], errors='coerce')
        data = data.dropna()
        return data
    except Exception as e:
        print(f"Greška prilikom pripreme ulaznih podataka: {e}")
        raise
