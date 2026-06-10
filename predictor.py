import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Configuración de la API Key usando el nombre exacto de tus Secrets de GitHub
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
SPORT_KEY = "soccer_mexico_liga_mx"  # Cambiar por liga de preferencia (ej: soccer_conmebol_copa_libertadores)
REGIONS = "us,eu"                    # Región de las casas de apuestas
MARKETS = "h2h"                      # Mercado de resultado directo (1X2)

def obtener_partidos_espn():
    """Extrae la cartelera de próximos partidos desde la API pública de ESPN"""
    print("-> Consultando API de ESPN...")
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard"
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
            
            # Identificar local y visitante según la propiedad de ESPN
            try:
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
        print(f"Advertencia: No se pudo conectar a ESPN ({e}). Cargando cartelera base por defecto.")
        return []

def obtener_cuotas_the_odds():
    """Extrae las cuotas vigentes del mercado usando ODDS_API_KEY"""
    if not ODDS_API_KEY:
        print("-> Alerta: ODDS_API_KEY no configurada en el entorno. Saltando consulta de cuotas.")
        return {}
        
    print("-> Consultando The Odds API...")
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
                    # Extraer cuotas del primer operador disponible en la lista
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
                            
                    # Mapear con llave en minúsculas para mitigar variaciones ortográficas
                    key = f"{home_team.lower()}_{away_team.lower()}"
                    cuotas_mapeadas[key] = odds_dict
            return cuotas_mapeadas
        else:
            print(f"Error en API de Cuotas (Status {response.status_code}): {response.text}")
            return {}
    except Exception as e:
        print(f"Error de conexión con la API de cuotas: {e}")
        return {}

def ejecutar_sistema_prediccion():
    print("=== [BGLogicSolutions] Iniciando Pipeline de Predicción Analítica ===")

    # 1. BASE DE DATOS HISTÓRICA DE ENTRENAMIENTO
    data_historica = {
        'home_team': ['América', 'Guadalajara', 'Cruz Azul', 'Tigres', 'Monterrey', 'Pumas', 'León', 'América', 'Guadalajara', 'Cruz Azul', 'Toluca', 'Pachuca'],
        'away_team': ['Guadalajara', 'Cruz Azul', 'Tigres', 'Monterrey', 'Pumas', 'León', 'América', 'Tigres', 'Monterrey', 'Pumas', 'América', 'Guadalajara'],
        'home_score': [2, 1, 0, 3, 1, 0, 2, 1, 3, 2, 1, 1],
        'away_score': [1, 1, 1, 0, 2, 2, 1, 0, 1, 2, 2, 0],
        'neutral': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data_historica)

    # 2. OBTENER CARTELERA Y CUOTAS EN VIVO
    partidos_nuevos = obtener_partidos_espn()
    if not partidos_nuevos:
        # Cartelera de contingencia si no hay partidos programados en vivo hoy en ESPN
        partidos_nuevos = [
            {"local": "América", "visitante": "Guadalajara", "neutral": 0},
            {"local": "Cruz Azul", "visitante": "Tigres", "neutral": 0},
            {"local": "Monterrey", "visitante": "Pumas", "neutral": 0}
        ]
    
    cuotas_reales = obtener_cuotas_the_odds()

    # 3. CODIFICACIÓN ALFANUMÉRICA (Label Encoding)
    le = LabelEncoder()
    todos_los_equipos = list(set(
        df['home_team'].tolist() + df['away_team'].tolist() + 
        [p['local'] for p in partidos_nuevos] + [p['visitante'] for p in partidos_nuevos]
    ))
    le.fit(todos_los_equipos)
    
    df['home_encoded'] = le.transform(df['home_team'])
    df['away_encoded'] = le.transform(df['away_team'])

    # 4. ENTRENAMIENTO DE LOS ALGORITMOS REPLICADOS
    X = df[['home_encoded', 'away_encoded', 'neutral']]
    
    # Modelos Random Forest
    rf_home = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['home_score'])
    rf_away = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['away_score'])
    
    # Modelos SVR
    svr_home = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, df['home_score'])
    svr_away = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, df['away_score'])

    # 5. CONSTRUCCIÓN DEL REPORTE
    lineas_reporte = [
        "==========================================================",
        "  REPORTE INTEGRADO: MODELOS IA + CUOTAS EN VIVO (ESPN)   ",
        "==========================================================\n"
    ]

    for partido in partidos_nuevos:
        loc = partido["local"]
        vis = partido["visitante"]
        neu = partido["neutral"]
        
        id_l = le.transform([loc])[0]
        id_v = le.transform([vis])[0]
        input_data = pd.DataFrame([[id_l, id_v, neu]], columns=['home_encoded', 'away_encoded', 'neutral'])
        
        # Inferencia y redondeo matemático de marcadores
        g_l_rf = int(np.round(rf_home.predict(input_data)[0]))
        g_v_rf = int(np.round(rf_away.predict(input_data)[0]))
        
        g_l_svr = int(np.round(svr_home.predict(input_data)[0]))
        g_v_svr = int(np.round(svr_away.predict(input_data)[0]))

        # Match de cuotas por nombre en minúsculas para mitigar variaciones ortográficas de las APIs
        key_busqueda = f"{loc.lower()}_{vis.lower()}"
        cuotas = cuotas_reales.get(key_busqueda, {"cuota_local": "N/D", "cuota_empate": "N/D", "cuota_visitante": "N/D"})

        lineas_reporte.append(f"PARTIDO: {loc} vs {vis}")
        lineas_reporte.append(f"  [Random Forest] Predicción: {g_l_rf} - {g_v_rf}")
        lineas_reporte.append(f"  [SVR Machine]  Predicción: {g_l_svr} - {g_v_svr}")
        lineas_reporte.append(f"  [Cuotas Odds API] Local: {cuotas['cuota_local']} | Empate: {cuotas['cuota_empate']} | Visitante: {cuotas['cuota_visitante']}")
        lineas_reporte.append("-" * 55)

    # 6. PERSISTENCIA EN ARCHIVO LOCAL
    with open("predicciones_reporte.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_reporte))
    print("-> Archivo 'predicciones_reporte.txt' sobrescrito exitosamente.")

if __name__ == "__main__":
    ejecutar_sistema_prediccion()
