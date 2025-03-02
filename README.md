# BDUIS: Predikcija Solarne Energije

## Opis projekta
**BDUIS** je aplikacija razvijena za predikciju proizvodnje solarne energije na tri različite lokacije, koristeći vremenske podatke i solarne parametre.  
Aplikacija omogućava korisnicima da analiziraju **satne** predikcije proizvodnje (predikcija za dan i mjesec je uklonjena jer nema dovoljno podataka za analizu), vizualizuju uticaj različitih parametara, kao i da dodaju nove podatke u postojeći skup podataka.

---

## Funkcionalnosti

### 1. **Predikcija proizvodnje solarne energije**
- Korisnici mogu birati **satni interval** predikcije.
- Predikcije su prikazane u vidu interaktivnih grafova sa procentualnim prikazom doprinosa svake lokacije.
- Prosečna vrednost proizvodnje za lokacije prikazana je kao linija na grafu.

### 2. **Vizualizacija uticaja parametara**
- Za svaku lokaciju prikazana je važnost solarnih i vremenskih parametara na proizvodnju.
- Grafovi prikazuju:
  - Sortirane parametre prema važnosti.
  - Procentualni doprinos svakog parametra.
  - Prosečnu važnost kroz horizontalnu liniju.

### 3. **Dodavanje novih podataka**
- Korisnici mogu dodavati nove podatke u CSV fajl putem jednostavne forme.
- Implementirane su validacije unosa za sve parametre kako bi se osigurali ispravni podaci.
- Podaci se odmah integrišu u postojeći dataset za dalju analizu.

---

## Tehnologije
- **Programski jezik**: Python
- **Biblioteke**:
  - Flask (za backend i frontend integraciju)
  - Plotly (za vizualizaciju grafova)
  - Scikit-learn (za modele predikcije)
  - Pandas (za obradu podataka)
- **Frontend**:
  - HTML/CSS (Bootstrap za stilizaciju)
- **Baza podataka**: CSV format za čuvanje i obradu podataka.

---

## Instalacija
1. Klonirajte repozitorijum:
   ```bash
   git clone https://github.com/MastilovicRadoslav/BDUIS.git
2. Pređite u direktorijum projekta:
   cd BDUIS
3. Kreirajte i aktivirajte virtuelno okruženje:
   python -m venv venv
   source venv/bin/activate  # Na Windows-u: venv\Scripts\activate
4. Instalirajte zavisnosti:
   pip install -r requirements.txt
5. Pokrenite aplikaciju:
   python main.py
6. Otvorite aplikaciju u svom pretraživaču na adresi:
   http://127.0.0.1:5000/
---

## Struktura projekta
![1](https://github.com/user-attachments/assets/5e7e6b0e-7280-48ba-ad8e-7d4ff93e3380)

---

# Uputstvo za korišćenje

## Početna stranica
1. Izaberite interval za predikciju (*Sat*)
2. Vizualizujte rezultate kroz grafove i liste.

## Dodavanje novih podataka
1. Kliknite na dugme **Dodaj nove podatke**.
2. Popunite formu sa potrebnim parametrima.
3. Kliknite na dugme **Dodaj podatke** za čuvanje unosa.

## Primeri korišćenja

### Primer predikcije
1. Izaberite interval **Sat**.
2. Pregledajte predikciju proizvodnje za svaku lokaciju i ukupnu vrednost.
3. Analizirajte graf važnosti parametara.

### Dodavanje novih podataka
1. Unesite datum i vreme u formatu `MM/DD/YYYY HH:MM`.
2. Dodajte vrednosti za sve potrebne parametre.
3. Potvrdite unos i vratite se na početnu stranicu.

## Autor
**Radoslav Mastilović**

## Planirani dodaci
- Migracija baze podataka na SQL za efikasnije upravljanje.
- Implementacija API-ja za integraciju sa eksternim sistemima.
- Poboljšana analitika kroz dodatne metrike i vizualizacije.

## Video demonstracija 
Pogledajte demonstraciju aplikacije: [YouTube](https:https://www.youtube.com/watch?v=jIpKy8Na2_g&ab_channel=%D0%A0%D0%B0%D0%B4%D0%BE%D1%81%D0%BB%D0%B0%D0%B2%D0%9C%D0%B0%D1%81%D1%82%D0%B8%D0%BB%D0%BE%D0%B2%D0%B8%D1%9B)

## Licenca
Ovaj projekat je otvorenog koda i dostupan pod **MIT licencom**.

