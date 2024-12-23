from flask import Flask, request, render_template
from models.train_model import (model_location_1, model_location_2, model_location_3)
from models.data_preprocessing import load_and_preprocess_data
import pandas as pd
import plotly.graph_objects as go

app = Flask(__name__)

# Učitavanje podataka
X, _ = load_and_preprocess_data('dataset/Data_Cacak.csv')

# Generisanje grafa predikcija
def generate_chart(predictions, selected_total, label):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Lokacija 1', 'Lokacija 2', 'Lokacija 3'], y=predictions, name='Predikcija po lokacijama'))
    fig.add_trace(go.Bar(x=['Ukupno'], y=[selected_total], name=label, marker_color='red'))
    fig.update_layout(title='Predikcija solarne energije', xaxis_title='Kategorija', yaxis_title='Proizvodnja (kWh)')
    return fig.to_html(full_html=False)

# Generisanje grafa feature importances
def generate_feature_importance_chart(importances, feature_names, title):
    fig = go.Figure([go.Bar(x=feature_names, y=importances)])
    fig.update_layout(title=title, xaxis_title="Parametri", yaxis_title="Važnost parametra")
    return fig.to_html(full_html=False)

@app.route('/', methods=['GET', 'POST'])
@app.route('/predict/<interval>', methods=['GET', 'POST'])
def predict(interval="hourly"):
    result = None
    chart_html = ""
    feature_importance_htmls = []
    interval_class = interval  # Postavlja klasu intervala na osnovu trenutnog intervala

    if request.method == 'POST':
        try:
            # Priprema podataka za predikcije
            input_data = pd.DataFrame([X.iloc[-1].tolist()], columns=X.columns)
            models = [model_location_1, model_location_2, model_location_3]

            # Predikcije za sve lokacije
            predictions = [model.predict(input_data)[0] for model in models]
            
            if interval == "hourly":
                selected_predictions = predictions
                selected_total = sum(predictions)
                label = "Ukupno - satna predikcija"
            elif interval == "daily":
                selected_predictions = [p * 24 for p in predictions]  # Predikcija za 24 sata
                selected_total = sum(selected_predictions)
                label = "Ukupno - dnevna predikcija"
            elif interval == "monthly":
                selected_predictions = [p * 24 * 30 for p in predictions]  # Predikcija za 30 dana
                selected_total = sum(selected_predictions)
                label = "Ukupno - mesečna predikcija"
            else:
                raise ValueError("Nepoznat interval")

            # Rezultati predikcije
            result = {
                'location_1': selected_predictions[0],
                'location_2': selected_predictions[1],
                'location_3': selected_predictions[2],
                'total_production': selected_total
            }

            # Glavni graf
            chart_html = generate_chart(selected_predictions, selected_total, label)

            # Grafovi važnosti parametara za sve lokacije
            feature_importance_htmls = [
                generate_feature_importance_chart(
                    model.named_steps['regressor'].feature_importances_, input_data.columns, f"Uticaj parametara - Lokacija {i+1}"
                ) for i, model in enumerate(models)
            ]

            # Graf za ukupnu proizvodnju (prosečan uticaj parametara)
            avg_feature_importances = sum(
                model.named_steps['regressor'].feature_importances_ for model in models
            ) / len(models)
            total_feature_importance_chart = generate_feature_importance_chart(
                avg_feature_importances, input_data.columns, "Uticaj parametara - Ukupna proizvodnja"
            )
            feature_importance_htmls.append(total_feature_importance_chart)

        except Exception as e:
            result = f"Greška: {e}"

    return render_template(
        "index.html",
        result=result,
        chart_html=chart_html,
        feature_importance_htmls=feature_importance_htmls,
        interval_class=interval_class
    )


if __name__ == "__main__":
    app.run(debug=True)
