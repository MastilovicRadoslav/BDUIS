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

# Generisanje grafa
def generate_prediction_chart(predictions, total_prediction):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Lokacija 1', 'Lokacija 2', 'Lokacija 3'], y=predictions, name='Predikcija po lokaciji'))
    fig.add_trace(go.Bar(x=['Ukupno'], y=[total_prediction], name='Predikcija ukupnog modela'))
    fig.update_layout(title='Predikcija solarne energije', xaxis_title='Lokacije', yaxis_title='Proizvodnja (kWh)')
    return fig.to_html(full_html=False)

# Glavna ruta za predikciju
@app.route('/', methods=['GET', 'POST'])
@app.route('/predict/<interval>', methods=['GET', 'POST'])
def predict(interval="hourly"):
    result = None
    chart_html = ""
    if request.method == 'POST':
        try:
            if interval == "hourly":
                input_data = pd.DataFrame([X.iloc[-1].tolist()], columns=X.columns)
                models = [model_location_1, model_location_2, model_location_3]
                total_model = model_total
            elif interval == "daily":
                input_data = pd.DataFrame([X_daily.iloc[-1].tolist()], columns=X_daily.columns)
                models = [model_daily_1, model_daily_2, model_daily_3]
                total_model = model_daily_total
            elif interval == "monthly":
                input_data = pd.DataFrame([X_monthly.iloc[-1].tolist()], columns=X_monthly.columns)
                models = [model_monthly_1, model_monthly_2, model_monthly_3]
                total_model = model_monthly_total
            else:
                return "Greška: Nepostojeći interval!"

            # Predikcije
            predictions = [model.predict(input_data)[0] for model in models]
            total_prediction = total_model.predict(input_data)[0]
            sum_prediction = sum(predictions)
            result = {
                'location_1': predictions[0],
                'location_2': predictions[1],
                'location_3': predictions[2],
                'sum_total': sum_prediction,
                'model_total': total_prediction
            }
            chart_html = generate_prediction_chart(predictions, total_prediction)

        except Exception as e:
            result = f"Greška: {e}"

    return render_template("index.html", result=result, chart_html=chart_html)

if __name__ == "__main__":
    app.run(debug=True)
