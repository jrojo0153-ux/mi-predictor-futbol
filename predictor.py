import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# =========================================================================
# CONFIGURACIÓN EXACTA PARA PARTIDOS AMISTOSOS RUMBO AL MUNDIAL 2026
# =========================================================================
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
SPORT_KEY = "soccer_international_friendly"  # ID oficial para amistosos internacionales
REGIONS = "us,eu"                             # Casas de apuestas
MARKETS = "h2h"                               # Ganador / Empate / Visitante

def obtener_partidos_espn():
    """Extrae la cartelera de partidos amistosos internacionales desde la API de ESPN"""
    print("-> Consultando API de ESPN (Amistosos Internacionales)...")
    # URL específica para amistosos de selecciones (International Friendly)
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.friendly/scoreboard"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        partidos = []
        
        for event in data.get('events', []):
            competitions = event.get('competitions', [])
            if not competitions:
                continue
            competitors = competitions[0].get('competitors', [])
            
            try:
                # Detectar local y visitante
                home = next(c['team']['displayName'] for c in competitors if c.get('homeAway') == 'home')
                away = next(c['team']['displayName'] for c in competitors if c.get('homeAway') == 'away')
                neutral = 1 if competitions[0].get('neutralGames', False) else 0
                
                partidos.append({
                    "local": home,
                    "visitante": away,
                    "neutral": neutral
                })
            except StopIteration:
                continue
        return partidos
    except Exception as e:
        print(f"Advertencia: No se pudo conectar a ESPN ({e}). Cargando amistosos de prueba.")
        return []

def obtener_cuotas_the_odds():
    """Extrae las cuotas vigentes para amistosos usando ODDS_API_KEY"""
    if not ODDS_API_KEY:
        print("-> Alerta: ODDS_API_KEY no configurada. Saltando consulta de cuotas.")
        return {}
        
    print(f"-> Consultando The Odds API para: {SPORT_KEY}...")
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': 'decimal'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            odds_data = response.json()
            cuotas_mapeadas = {}
            
            for match in odds_data:
                home_team = match.get('home_team', '')
                away_team = match.get('away_team', '')
                bookmakers = match.get('bookmakers', [])
                
                if bookmakers:
                    market = bookmakers[0].get('markets', [])[0]
                    outcomes = market.get('outcomes', [])
                    
                    odds_dict = {"cuota_local": "N/D", "cuota_empate": "N/D", "cuota_visitante": "N/D"}
                    for outcome in outcomes:
                        name = outcome.get('name', '')
                        price = outcome.get('price', 'N/D')
                        if name == home_team:
                            odds_dict['cuota_local'] = price
                        elif name == away_team:
                            odds_dict['cuota_visitante'] = price
                        elif name.lower() in ['draw', 'empate']:
                            odds_dict['cuota_empate'] = price
                            
                    # Llave de cruce en minúsculas
                    key = f"{home_team.lower()}_{away_team.lower()}"
                    cuotas_mapeadas[key] = odds_dict
            return cuotas_mapeadas
        elif response.status_code == 404:
            print(f"-> La clave '{SPORT_KEY}' no devolvió mercados activos en este instante.")
            return {}
        else:
            print(f"Error en API (Status {response.status_code}): {response.text}")
            return {}
    except Exception as e:
        print(f"Error de conexión con The Odds API: {e}")
        return {}

def ejecutar_sistema_prediccion():
    print("=== [BGLogicSolutions] Pipeline de Amistosos Internacionales ===")

    # 1. HISTÓRICO DE ENTRENAMIENTO (Partidos de preparación previos)
    # Puedes ampliar este diccionario con los resultados reales de los últimos amistosos
    data_historica = {
        'home_team': ['México', 'Estados Unidos', 'Argentina', 'Brasil', 'Colombia', 'España', 'Alemania', 'México', 'Francia', 'Argentina'],
        'away_team': ['Estados Unidos', 'Colombia', 'Brasil', 'México', 'Alemania', 'Brasil', 'Francia', 'Argentina', 'Chile', 'Ecuador'],
        'home_score': [1, 1, 2, 0, 3, 0, 1, 2, 3, 1],
        'away_score': [2, 5, 0, 1, 2, 1, 2, 0, 2, 0],
        'neutral': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data_historica)

    # 2. CONSUMO DE DATOS EN VIVO
    partidos_nuevos = obtener_partidos_espn()
    if not partidos_nuevos:
        # Cartelera de respaldo por si no hay amistosos jugándose en las próximas horas
        partidos_nuevos = [
            {"local": "México", "visitante": "Estados Unidos", "neutral": 1},
            {"local": "Argentina", "visitante": "Brasil", "neutral": 1},
            {"local": "Francia", "visitante": "Alemania", "neutral": 0}
        ]
    
    cuotas_reales = obtener_cuotas_the_odds()

    # 3. LABEL ENCODING GENERALE DE SELECCIONES
    le = LabelEncoder()
    todas_las_selecciones = list(set(
        df['home_team'].tolist() + df['away_team'].tolist() + 
        [p['local'] for p in partidos_nuevos] + [p['visitante'] for p in partidos_nuevos]
    ))
    le.fit(todas_las_selecciones)
    
    df['home_encoded'] = le.transform(df['home_team'])
    df['away_encoded'] = le.transform(df['away_team'])

    # 4. ENTRENAMIENTO DE LOS MODELOS REPLICADOS DEL VIDEO
    X = df[['home_encoded', 'away_encoded', 'neutral']]
    
    rf_home = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['home_score'])
    rf_away = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['away_score'])
    
    svr_home = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, df['home_score'])
    svr_away = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, df['away_score'])

    # 5. GENERACIÓN DEL REPORTE
    lineas_reporte = [
        "==========================================================",
        " REPORTE: AMISTOSOS RUMBO AL MUNDIAL 2026 (IA + ODDS)    ",
        "==========================================================\n"
    ]

    for partido in partidos_nuevos:
        loc = partido["local"]
        vis = partido["visitante"]
        neu = partido["neutral"]
        
        id_l = le.transform([loc])[0]
        id_v = le.transform([vis])[0]
        input_data = pd.DataFrame([[id_l, id_v, neu]], columns=['home_encoded', 'away_encoded', 'neutral'])
        
        # Inferencia matemática
        g_l_rf = int(np.round(rf_home.predict(input_data)[0]))
        g_v_rf = int(np.round(rf_away.predict(input_data)[0]))
        
        g_l_svr = int(np.round(svr_home.predict(input_data)[0]))
        g_v_svr = int(np.round(svr_away.predict(input_data)[0]))

        # Mapeo de nombres para emparejar las cuotas de las apuestas
        key_busqueda = f"{loc.lower()}_{vis.lower()}"
        cuotas = cuotas_reales.get(key_busqueda, {"cuota_local": "N/D", "cuota_empate": "N/D", "cuota_visitante": "N/D"})

        lineas_reporte.append(f"PARTIDO: {loc} vs {vis} (Neutral: {'Sí' if neu==1 else 'No'})")
        lineas_reporte.append(f"  [Random Forest] Predicción Goles: {g_l_rf} - {g_v_rf}")
        lineas_reporte.append(f"  [SVR Machine]  Predicción Goles: {g_l_svr} - {g_v_svr}")
        lineas_reporte.append(f"  [Cuotas Mercado] Local: {cuotas['cuota_local']} | Empate: {cuotas['cuota_empate']} | Visitante: {cuotas['cuota_visitante']}")
        lineas_reporte.append("-" * 55)

    with open("predicciones_reporte.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_reporte))
    print("-> Archivo 'predicciones_reporte.txt' actualizado sin errores.")

if __name__ == "__main__":
    ejecutar_sistema_prediccion()
