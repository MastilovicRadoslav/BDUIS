from flask import Flask, request, render_template
from models.train_model import (model_location_1, model_location_2, model_location_3)
from models.data_preprocessing import load_and_preprocess_data
import pandas as pd
import plotly.graph_objects as go

app = Flask(__name__)

# Učitavanje i priprema podataka
# Ovaj deo se izvršava jednom pri pokretanju aplikacije
X, _ = load_and_preprocess_data('dataset/Data_Cacak.csv')

#Generisanje grafa za predikciju proizvodnje
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
        text=[f"Prosek: {average_prediction:.2f} kWh"],  # Tekst na liniji
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


#Generisanje grafa za uticaj parametara na proizvodnju
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
        yaxis=dict(
            ticksuffix='%', 
            range=[0, sorted_importances.max() * 1.2]  # Proširuje opseg y-osi za 20% iznad najveće vrednosti
        ),
        xaxis=dict(tickangle=45)  # Rotacija oznaka na X-osi za bolju čitljivost
    )

    return fig.to_html(full_html=False)



#Glavna ruta za predikcije. Obrada intervala (sat, dan, mesec).
@app.route('/', methods=['GET', 'POST'])
@app.route('/predict/<interval>', methods=['GET', 'POST'])
def predict(interval="hourly"):
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

#Dugme za dodavanje novog mjerenja
@app.route('/add', methods=['GET', 'POST'])
def add_data():
    if request.method == 'POST':
        try:
            # Učitavanje unosa iz forme
            datetime = request.form['datetime']
            air_temp = float(request.form['air_temp'])
            cloud_opacity = float(request.form['cloud_opacity'])
            dhi = float(request.form['dhi'])
            dni = float(request.form['dni'])
            ebh = float(request.form['ebh'])
            ghi = float(request.form['ghi'])
            prod_loc1 = float(request.form['prod_loc1'])
            prod_loc2 = float(request.form['prod_loc2'])
            prod_loc3 = float(request.form['prod_loc3'])

            # Validacija unosa
            # Datum i vreme (proverava ispravan format)
            try:
                pd.to_datetime(datetime, format='%m/%d/%Y %H:%M')
            except ValueError:
                raise ValueError("Datum i vreme moraju biti u formatu MM/DD/YYYY HH:MM.")
            # Temperatura vazduha
            if air_temp < -50 or air_temp > 50:
                raise ValueError("Temperatura mora biti između -50 i 50 stepeni.")
            # Oblačnost
            if cloud_opacity < 0 or cloud_opacity > 100:
                raise ValueError("Oblačnost mora biti između 0% i 100%.")
            # DHI
            if dhi < 0:
                raise ValueError("DHI mora biti pozitivan broj.")
            # DNI
            if dni < 0:
                raise ValueError("DNI mora biti pozitivan broj.")
            # EBH
            if ebh < 0:
                raise ValueError("EBH mora biti pozitivan broj.")
            # GHI
            if ghi < 0:
                raise ValueError("GHI mora biti pozitivan broj.")
            # Proizvodnja - Lokacija 1
            if prod_loc1 < 0:
                raise ValueError("Proizvodnja za Lokaciju 1 mora biti pozitivan broj.")
            # Proizvodnja - Lokacija 2
            if prod_loc2 < 0:
                raise ValueError("Proizvodnja za Lokaciju 2 mora biti pozitivan broj.")
            # Proizvodnja - Lokacija 3
            if prod_loc3 < 0:
                raise ValueError("Proizvodnja za Lokaciju 3 mora biti pozitivan broj.")
            # Dodavanje validiranih podataka u CSV
            with open('dataset/Data_Cacak.csv', 'a') as f:
                f.write(f"{datetime},{air_temp},{cloud_opacity},{dhi},{dni},{ebh},{ghi},{prod_loc1},{prod_loc2},{prod_loc3}\n")

            return "Podaci su uspešno dodati!"

        except Exception as e:
            return f"Greška: {e}"

    return render_template('add_data.html')


if __name__ == "__main__":
    app.run(debug=True)
