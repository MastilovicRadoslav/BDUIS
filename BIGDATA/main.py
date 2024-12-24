from flask import Flask, request, render_template
from models.train_model import (model_location_1, model_location_2, model_location_3)
from models.data_preprocessing import load_and_preprocess_data
import pandas as pd
import plotly.graph_objects as go

app = Flask(__name__)

# Učitavanje i priprema podataka
# Ovaj deo se izvršava jednom pri pokretanju aplikacije
X, _ = load_and_preprocess_data('dataset/Data_Cacak.csv')

def generate_chart(predictions, selected_total, label):
    """
    Kreira unapređeni bar graf za predikcije po lokacijama i ukupnu proizvodnju.
    - Dodavanje razmaka između stubića i oznaka.
    - Legenda pomerena ispod grafa.
    - Dodata linija za prosečnu predikciju proizvodnje sa vidljivim nazivom.
    """
    percentages = [(pred / selected_total) * 100 if selected_total > 0 else 0 for pred in predictions]
    average_prediction = sum(predictions) / len(predictions) if predictions else 0

    fig = go.Figure()
    
    # Dodavanje barova za predikcije po lokacijama
    fig.add_trace(go.Bar(
        x=['Lokacija 1', 'Lokacija 2', 'Lokacija 3'],
        y=predictions,
        text=[f"{pred:.2f} kWh<br>({perc:.2f}%)" for pred, perc in zip(predictions, percentages)],
        textposition='outside',
        marker_color=['#3498db', '#2ecc71', '#e74c3c'],
        name='Predikcija po lokacijama'
    ))

    # Dodavanje bara za ukupnu proizvodnju
    fig.add_trace(go.Bar(
        x=['Ukupno'],
        y=[selected_total],
        text=f"{selected_total:.2f} kWh",
        textposition='outside',
        marker_color='red',
        name=label
    ))

    # Dodavanje linije za prosečnu predikciju
    fig.add_trace(go.Scatter(
        x=['Lokacija 1', 'Lokacija 2', 'Lokacija 3'],
        y=[average_prediction] * 3,
        mode='lines+text',
        name='Prosečna proizvodnja',
        line=dict(color='orange', dash='dash'),
        text=[f"Prosečna predikcija za proizvodnju: {average_prediction:.2f} kWh"],  # Tekst na liniji
        textposition="top center"  # Pozicija teksta
    ))

    # Podešavanje izgleda grafa
    fig.update_layout(
        title='Predikcija solarne energije',
        xaxis_title='Kategorija',
        yaxis_title='Proizvodnja (kWh)',
        yaxis=dict(ticksuffix=' kWh', range=[0, max(predictions + [selected_total]) * 1.2]),
        xaxis=dict(tickangle=0),
        barmode='group',
        legend=dict(
            title='Legenda',
            orientation='h',  # Horizontalna legenda
            y=-0.2,  # Pozicija ispod grafa
            x=0.5,
            xanchor='center',
            font=dict(size=10)  # Smanjen font za legendu
        )
    )

    return fig.to_html(full_html=False)



def generate_feature_importance_chart(importances, feature_names, title):
    """
    Generiše graf važnosti parametara sa procentima, sortiranjem i unapređenjima:
    - Sortirani parametri po važnosti
    - Dinamičke boje na osnovu važnosti
    - Tekst sa procentima iznad stubića
    - Horizontalna linija prosečne važnosti
    - Interaktivni hover sa dodatnim informacijama
    """
    # Konvertovanje sirovih vrednosti u procente
    importance_percentages = (importances / importances.sum()) * 100

    # Sortiranje parametara po važnosti
    sorted_indices = importance_percentages.argsort()[::-1]
    sorted_importances = importance_percentages[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]

    # Dinamičke boje na osnovu važnosti (više važnosti = tamnije boje)
    colors = ['#2ecc71' if val > 20 else '#3498db' if val > 10 else '#e74c3c' for val in sorted_importances]

    # Kreiranje grafa
    fig = go.Figure([go.Bar(
        x=sorted_feature_names,
        y=sorted_importances,
        marker_color=colors,
        text=[f"{val:.2f}%" for val in sorted_importances],  # Prikaz vrednosti sa procentima
        textposition='outside',  # Pozicija teksta iznad stubića
        hovertext=[f"Parametar: {name}<br>Važnost: {val:.2f}%" for name, val in zip(sorted_feature_names, sorted_importances)],
        hoverinfo="text"  # Interaktivni hover sa dodatnim informacijama
    )])

    # Dodavanje horizontalne linije prosečne važnosti
    average_importance = sorted_importances.mean()
    fig.add_hline(
        y=average_importance,
        line_dash="dash",
        line_color="black",
        annotation_text="Prosečna važnost",
        annotation_position="top left"
    )

    # Podešavanje izgleda grafa
    fig.update_layout(
        title=title,
        xaxis_title="Parametri (sortirani)",
        yaxis_title="Važnost parametara (%)",
        yaxis=dict(ticksuffix='%'),
        xaxis=dict(tickangle=45)  # Rotacija oznaka na X-osi za bolju čitljivost
    )

    return fig.to_html(full_html=False)



@app.route('/', methods=['GET', 'POST'])
@app.route('/predict/<interval>', methods=['GET', 'POST'])
def predict(interval="hourly"):
    """
    Glavna ruta za predikcije. Obrada intervala (sat, dan, mesec).
    """
    result = None
    chart_html = ""
    feature_importance_htmls = []
    interval_class = interval  # Klasa za stilizaciju aktivnog dugmeta

    if request.method == 'POST':
        try:
            # Priprema ulaznih podataka za modele
            input_data = pd.DataFrame([X.iloc[-1].tolist()], columns=X.columns)
            models = [model_location_1, model_location_2, model_location_3]

            # Izračunavanje predikcija
            predictions = [model.predict(input_data)[0] for model in models]
            
            # Obrada na osnovu izabranog intervala
            if interval == "hourly":
                selected_predictions = predictions
                selected_total = sum(predictions)
                label = "Ukupno - satna predikcija"
            elif interval == "daily":
                selected_predictions = [p * 24 for p in predictions]  # 24 sata
                selected_total = sum(selected_predictions)
                label = "Ukupno - dnevna predikcija"
            elif interval == "monthly":
                selected_predictions = [p * 24 * 30 for p in predictions]  # 30 dana
                selected_total = sum(selected_predictions)
                label = "Ukupno - mesečna predikcija"
            else:
                raise ValueError("Nepoznat interval")

            # Rezultati predikcije za prikaz
            result = {
                'location_1': selected_predictions[0],
                'location_2': selected_predictions[1],
                'location_3': selected_predictions[2],
                'total_production': selected_total
            }

            # Generisanje glavnog grafa predikcija
            chart_html = generate_chart(selected_predictions, selected_total, label)

            # Generisanje grafova važnosti parametara
            feature_importance_htmls = [
                generate_feature_importance_chart(
                    model.named_steps['regressor'].feature_importances_, input_data.columns, f"Uticaj parametara - Lokacija {i+1}"
                ) for i, model in enumerate(models)
            ]

            # Graf za prosečan uticaj parametara (ukupna proizvodnja)
            avg_feature_importances = sum(
                model.named_steps['regressor'].feature_importances_ for model in models
            ) / len(models)
            total_feature_importance_chart = generate_feature_importance_chart(
                avg_feature_importances, input_data.columns, "Uticaj parametara - Ukupna proizvodnja"
            )
            feature_importance_htmls.append(total_feature_importance_chart)

        except Exception as e:
            # Ako dođe do greške, proslediti poruku
            result = f"Greška: {e}"

    # Renderovanje šablona sa rezultatima i grafovima
    return render_template(
        "index.html",
        result=result,
        chart_html=chart_html,
        feature_importance_htmls=feature_importance_htmls,
        interval_class=interval_class
    )

if __name__ == "__main__":
    app.run(debug=True)
