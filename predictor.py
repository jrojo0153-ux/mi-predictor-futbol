import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# =========================================================================
# CONFIGURACIÓN OPTIMIZADA PARA AJUSTE DE CUOTAS
# =========================================================================
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
# Cambiamos a 'upcoming' para asegurar que la API devuelva cuotas reales de partidos HOY
SPORT_KEY = "upcoming"  
REGIONS = "us,eu"                    
MARKETS = "h2h"                      

def obtener_partidos_espn():
    """Extrae la cartelera actual desde la API de ESPN"""
    print("-> Consultando API de ESPN...")
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.friendly/scoreboard"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        partidos = []
        for event in data.get('events', []):
            competitions = event.get('competitions', [])
            if not competitions: continue
            competitors = competitions[0].get('competitors', [])
            try:
                home = next(c['team']['displayName'] for c in competitors if c.get('homeAway') == 'home')
                away = next(c['team']['displayName'] for c in competitors if c.get('homeAway') == 'away')
                neutral = 1 if competitions[0].get('neutralGames', False) else 0
                partidos.append({"local": home, "visitante": away, "neutral": neutral})
            except StopIteration: continue
        return partidos
    except Exception as e:
        print(f"Advertencia ESPN: {e}")
        return []

def obtener_cuotas_the_odds():
    """Extrae las cuotas vigentes del mercado e imprime diagnóstico en consola"""
    if not ODDS_API_KEY:
        print("-> Error: ODDS_API_KEY vacía en el entorno de ejecución.")
        return {}
        
    print(f"-> Conectando a The Odds API (Filtro: {SPORT_KEY})...")
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': 'decimal'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"-> Respuesta de API de Cuotas: Status {response.status_code}")
        
        if response.status_code == 200:
            odds_data = response.json()
            print(f"-> Éxito: Se recuperaron {len(odds_data)} partidos con cuotas activas en el mercado.")
            
            # Mostrar los primeros 2 partidos de la API en la consola de GitHub para auditar nombres
            if odds_data:
                print("--- AUDITORÍA DE NOMBRES EN LA API ---")
                for m in odds_data[:2]:
                    print(f" Disponible en API -> Local: '{m.get('home_team')}' vs Visitante: '{m.get('away_team')}'")
                print("--------------------------------------")
                
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
                        if name == home_team: odds_dict['cuota_local'] = price
                        elif name == away_team: odds_dict['cuota_visitante'] = price
                        elif name.lower() in ['draw', 'empate']: odds_dict['cuota_empate'] = price
                            
                    # Guardamos la llave normalizada
                    key = f"{home_team.lower()}_{away_team.lower()}"
                    cuotas_mapeadas[key] = odds_dict
            return cuotas_mapeadas
        else:
            print(f"-> Error devuelto por el servidor de cuotas: {response.text}")
            return {}
    except Exception as e:
        print(f"-> Fallo crítico de conexión con la API: {e}")
        return {}

def buscar_cuota_flexible(local, visitante, diccionario_cuotas):
    """Busca cuotas resolviendo variaciones idiomáticas básicas (ej: México vs Mexico)"""
    # 1. Intento de coincidencia exacta en minúsculas
    key_directa = f"{local.lower()}_{visitante.lower()}"
    if key_directa in diccionario_cuotas:
        return diccionario_cuotas[key_directa]
        
    # 2. Búsqueda por sub-cadena (tolerancia a traducciones como USA/Estados Unidos o tildes)
    for key, cuotas in diccionario_cuotas.items():
        api_local, api_visitante = key.split('_')
        # Si el nombre de ESPN está contenido en la API o viceversa
        if (local.lower() in api_local or api_local in local.lower()) or \
           (visitante.lower() in api_visitante or api_visitante in visitante.lower()):
            return cuotas
            
    return {"cuota_local": "N/D", "cuota_empate": "N/D", "cuota_visitante": "N/D"}

def ejecutar_sistema_prediccion():
    print("=== [BGLogicSolutions] Iniciando Pipeline de Predicción Analítica ===")

    # 1. HISTÓRICO DE ENTRENAMIENTO
    data_historica = {
        'home_team': ['México', 'Estados Unidos', 'Argentina', 'Brasil', 'Colombia', 'España', 'Alemania', 'México', 'Francia', 'Argentina'],
        'away_team': ['Estados Unidos', 'Colombia', 'Brasil', 'México', 'Alemania', 'Brasil', 'Francia', 'Argentina', 'Chile', 'Ecuador'],
        'home_score': [1, 1, 2, 0, 3, 0, 1, 2, 3, 1],
        'away_score': [2, 5, 0, 1, 2, 1, 2, 0, 2, 0],
        'neutral': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data_historica)

    # 2. CONSUMO DE DATOS
    partidos_nuevos = obtener_partidos_espn()
    cuotas_reales = obtener_cuotas_the_odds()

    # Si ESPN no tiene partidos hoy, forzamos una cartelera de prueba
    if not partidos_nuevos:
        print("-> Usando cartelera de simulación para el reporte.")
        partidos_nuevos = [
            {"local": "México", "visitante": "Estados Unidos", "neutral": 1},
            {"local": "Argentina", "visitante": "Brasil", "neutral": 1}
        ]

    # 3. PROCESAMIENTO MATEMÁTICO (Label Encoding)
    le = LabelEncoder()
    todas_las_selecciones = list(set(
        df['home_team'].tolist() + df['away_team'].tolist() + 
        [p['local'] for p in partidos_nuevos] + [p['visitante'] for p in partidos_nuevos]
    ))
    le.fit(todas_las_selecciones)
    df['home_encoded'] = le.transform(df['home_team'])
    df['away_encoded'] = le.transform(df['away_team'])

    # 4. ENTRENAMIENTO DE MODELOS
    X = df[['home_encoded', 'away_encoded', 'neutral']]
    rf_home = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['home_score'])
    rf_away = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['away_score'])
    svr_home = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, df['home_score'])
    svr_away = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, df['away_score'])

    # 5. CONSTRUCCIÓN DEL REPORTE
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
        
        g_l_rf = int(np.round(rf_home.predict(input_data)[0]))
        g_v_rf = int(np.round(rf_away.predict(input_data)[0]))
        g_l_svr = int(np.round(svr_home.predict(input_data)[0]))
        g_v_svr = int(np.round(svr_away.predict(input_data)[0]))

        # Aplicamos la nueva búsqueda flexible para evitar el bloqueo por idiomas o tildes
        cuotas = buscar_cuota_flexible(loc, vis, cuotas_reales)

        lineas_reporte.append(f"PARTIDO: {loc} vs {vis} (Neutral: {'Sí' if neu==1 else 'No'})")
        lineas_reporte.append(f"  [Random Forest] Predicción Goles: {g_l_rf} - {g_v_rf}")
        lineas_reporte.append(f"  [SVR Machine]  Predicción Goles: {g_l_svr} - {g_v_svr}")
        lineas_reporte.append(f"  [Cuotas Odds API] Local: {cuotas['cuota_local']} | Empate: {cuotas['cuota_empate']} | Visitante: {cuotas['cuota_visitante']}")
        lineas_reporte.append("-" * 55)

    with open("predicciones_reporte.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_reporte))
    print("-> Proceso completado con éxito. Reporte guardado.")

if __name__ == "__main__":
    ejecutar_sistema_prediccion()
