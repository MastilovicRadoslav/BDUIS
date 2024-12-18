from flask import Flask, request, render_template
from models.train_model import (model_location_1, model_location_2, model_location_3, model_total,
                                model_daily_1, model_daily_2, model_daily_3, model_daily_total,
                                model_monthly_1, model_monthly_2, model_monthly_3, model_monthly_total)
from models.data_preprocessing import load_and_preprocess_data, group_data_by_interval
import pandas as pd
import plotly.graph_objects as go

app = Flask(__name__)

# Učitavanje podataka
X, _ = load_and_preprocess_data('dataset/Data_Cacak.csv')
X_daily, _ = group_data_by_interval('dataset/Data_Cacak.csv', 'daily')
X_monthly, _ = group_data_by_interval('dataset/Data_Cacak.csv', 'monthly')

# Generisanje grafa predikcija
def generate_chart(predictions, selected_total, label):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Lokacija 1', 'Lokacija 2', 'Lokacija 3'], y=predictions, name='Predikcija po lokacijama'))
    fig.add_trace(go.Bar(x=['Ukupno'], y=[selected_total], name=label, marker_color='red'))
    fig.update_layout(title='Predikcija solarne energije', xaxis_title='Kategorija', yaxis_title='Proizvodnja (kWh)')
    return fig.to_html(full_html=False)

# Generisanje grafa feature importances
def generate_feature_importance_chart(model, feature_names, title):
    importance = model.named_steps['regressor'].feature_importances_
    fig = go.Figure([go.Bar(x=feature_names, y=importance)])
    fig.update_layout(title=title, xaxis_title="Parametri", yaxis_title="Važnost parametra")
    return fig.to_html(full_html=False)

@app.route('/', methods=['GET', 'POST'])
@app.route('/predict/<interval>', methods=['GET', 'POST'])
def predict(interval="hourly"):
    result = None
    chart_html = ""
    feature_importance_htmls = []
    interval_class = "hourly"  # Default pozadina

    if request.method == 'POST':
        try:
            if interval == "hourly":
                interval_class = "hourly"
                input_data = pd.DataFrame([X.iloc[-1].tolist()], columns=X.columns)
                models = [model_location_1, model_location_2, model_location_3]
                total_model = model_total
                selected_total = sum([model.predict(input_data)[0] for model in models])
                label = "Ukupno - sabiranje"

            elif interval == "daily":
                interval_class = "daily"
                input_data = pd.DataFrame([X_daily.iloc[-1].tolist()], columns=X_daily.columns)
                models = [model_daily_1, model_daily_2, model_daily_3]
                total_model = model_daily_total
                selected_total = total_model.predict(input_data)[0]
                label = "Ukupno - novi model"

            elif interval == "monthly":
                interval_class = "monthly"
                input_data = pd.DataFrame([X_monthly.iloc[-1].tolist()], columns=X_monthly.columns)
                models = [model_monthly_1, model_monthly_2, model_monthly_3]
                total_model = model_monthly_total
                selected_total = sum([model.predict(input_data)[0] for model in models])
                label = "Ukupno - sabiranje"

            # Predikcije
            predictions = [model.predict(input_data)[0] for model in models]
            result = {
                'location_1': predictions[0],
                'location_2': predictions[1],
                'location_3': predictions[2],
                'total_production': selected_total
            }

            # Glavni graf
            chart_html = generate_chart(predictions, selected_total, label)

            # Graftovi važnosti parametara
            feature_importance_htmls = [
                generate_feature_importance_chart(models[0], input_data.columns, "Uticaj parametara - Lokacija 1"),
                generate_feature_importance_chart(models[1], input_data.columns, "Uticaj parametara - Lokacija 2"),
                generate_feature_importance_chart(models[2], input_data.columns, "Uticaj parametara - Lokacija 3"),
                generate_feature_importance_chart(total_model, input_data.columns, "Uticaj parametara - Ukupna proizvodnja")
            ]

        except Exception as e:
            result = f"Greška: {e}"

    return render_template("index.html", result=result, chart_html=chart_html, feature_importance_htmls=feature_importance_htmls, interval_class=interval_class)

if __name__ == "__main__":
    app.run(debug=True)
