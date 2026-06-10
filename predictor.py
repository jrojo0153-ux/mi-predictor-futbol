import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# =========================================================================
# CONFIGURACIÓN COMPLETA - PIPELINE DE FÚTBOL AUTÓNOMO (BGLogicSolutions)
# =========================================================================
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# Usamos 'soccer' para forzar a la API a traer solo mercados de fútbol global,
# garantizando consistencia y evitando que se mezclen otros deportes como la MLB.
SPORT_KEY = "soccer"  
REGIONS = "us,eu"                    
MARKETS = "h2h" # Mercado Ganador / Empate / Visitante (Moneyline)

def obtener_partidos_espn():
    """Extrae la cartelera actual de partidos amistosos internacionales desde la API de ESPN"""
    print("-> Consultando API de ESPN (Amistosos Internacionales)...")
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
                # Detectar local, visitante y condición de campo
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
    """Extrae las cuotas vigentes de fútbol e imprime diagnóstico en la consola de GitHub"""
    if not ODDS_API_KEY:
        print("-> Alerta: ODDS_API_KEY vacía en el entorno de ejecución.")
        return {}
        
    print(f"-> Conectando a The Odds API (Filtro Deporte: {SPORT_KEY})...")
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
            print(f"-> Éxito: Se recuperaron {len(odds_data)} partidos de fútbol con cuotas activas.")
            
            # Auditoría diagnóstica en los logs del Workflow
            if odds_data:
                print("--- AUDITORÍA DE NOMBRES DE FÚTBOL EN LA API ---")
                for m in odds_data[:2]:
                    print(f" Disponible en API -> Local: '{m.get('home_team')}' vs Visitante: '{m.get('away_team')}'")
                print("------------------------------------------------")
                
            cuotas_mapeadas = {}
            for match in odds_data:
                home_team = match.get('home_team', '')
                away_team = match.get('away_team', '')
                bookmakers = match.get('bookmakers', [])
                
                if bookmakers:
                    # Extraer líneas del primer operador disponible del feed
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
                            
                    # Indexar usando minúsculas para unificar criterios de búsqueda
                    key = f"{home_team.lower()}_{away_team.lower()}"
                    cuotas_mapeadas[key] = odds_dict
            return cuotas_mapeadas
        else:
            print(f"-> Error de API de Cuotas ({response.status_code}): {response.text}")
            return {}
    except Exception as e:
        print(f"-> Fallo de conexión con el servidor de cuotas: {e}")
        return {}

def buscar_cuota_flexible(local, visitante, diccionario_cuotas):
    """Resuelve discrepancias de idioma, tildes o abreviaciones entre ESPN y las casas de apuestas"""
    # 1. Intento de cruce directo en minúsculas
    key_directa = f"{local.lower()}_{visitante.lower()}"
    if key_directa in diccionario_cuotas:
        return diccionario_cuotas[key_directa]
        
    # 2. Análisis flexible de sub-cadenas (Tolerancia a variaciones ej: México/Mexico o USA/Estados Unidos)
    for key, cuotas in diccionario_cuotas.items():
        api_local, api_visitante = key.split('_')
        if (local.lower() in api_local or api_local in local.lower()) or \
           (visitante.lower() in api_visitante or api_visitante in visitante.lower()):
            return cuotas
            
    return {"cuota_local": "N/D", "cuota_empate": "N/D", "cuota_visitante": "N/D"}

def ejecutar_sistema_prediccion():
    print("=== [BGLogicSolutions] Iniciando Pipeline de Predicción Analítica ===")

    # 1. HISTÓRICO DE ENTRENAMIENTO BASE (Márgenes e historial de goles de selecciones)
    data_historica = {
        'home_team': ['México', 'Estados Unidos', 'Argentina', 'Brasil', 'Colombia', 'España', 'Alemania', 'México', 'Francia', 'Argentina'],
        'away_team': ['Estados Unidos', 'Colombia', 'Brasil', 'México', 'Alemania', 'Brasil', 'Francia', 'Argentina', 'Chile', 'Ecuador'],
        'home_score': [1, 1, 2, 0, 3, 0, 1, 2, 3, 1],
        'away_score': [2, 5, 0, 1, 2, 1, 2, 0, 2, 0],
        'neutral': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data_historica)

    # 2. CAPTURA DE DATOS EN VIVO DESDE LAS APIS
    partidos_nuevos = obtener_partidos_espn()
    cuotas_reales = obtener_cuotas_the_odds()

    # Cartelera de respaldo/contingencia si no hay amistosos activos hoy en la API de ESPN
    if not partidos_nuevos:
        print("-> Usando cartelera de simulación para el reporte final.")
        partidos_nuevos = [
            {"local": "México", "visitante": "Estados Unidos", "neutral": 1},
            {"local": "Argentina", "visitante": "Brasil", "neutral": 1},
            {"local": "Francia", "visitante": "Alemania", "neutral": 0}
        ]

    # 3. PROCESAMIENTO MATEMÁTICO DE DATOS (Label Encoding)
    le = LabelEncoder()
    todas_las_selecciones = list(set(
        df['home_team'].tolist() + df['away_team'].tolist() + 
        [p['local'] for p in partidos_nuevos] + [p['visitante'] for p in partidos_nuevos]
    ))
    le.fit(todas_las_selecciones)
    
    df['home_encoded'] = le.transform(df['home_team'])
    df['away_encoded'] = le.transform(df['away_team'])

    # 4. ENTRENAMIENTO SIMULTÁNEO DE MODELOS DE REGRESIÓN
    X = df[['home_encoded', 'away_encoded', 'neutral']]
    
    # Modelos Random Forest Regressor
    rf_home = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['home_score'])
    rf_away = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['away_score'])
    
    # Modelos Support Vector Regressor (SVR)
    svr_home = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, df['home_score'])
    svr_away = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, df['away_score'])

    # 5. CONSTRUCCIÓN Y EXPORTACIÓN DEL REPORTE DE SALIDA
    lineas_reporte = [
        "==========================================================",
        " REPORTE: AMISTOSOS RUMBO AL MUNDIAL 2026 (IA + ODDS)    ",
        "==========================================================\n"
    ]

    for partido in partidos_nuevos:
        loc = partido["local"]
        vis = partido["visitante"]
        neu = partido["neutral"]
        
        # Mapear los equipos a sus códigos numéricos correspondientes
        id_l = le.transform([loc])[0]
        id_v = le.transform([vis])[0]
        input_data = pd.DataFrame([[id_l, id_v, neu]], columns=['home_encoded', 'away_encoded', 'neutral'])
        
        # Predicción continua redondeada al entero futbolístico más cercano
        g_l_rf = int(np.round(rf_home.predict(input_data)[0]))
        g_v_rf = int(np.round(rf_away.predict(input_data)[0]))
        
        g_l_svr = int(np.round(svr_home.predict(input_data)[0]))
        g_v_svr = int(np.round(svr_away.predict(input_data)[0]))

        # Ejecución del cruce adaptativo de cuotas
        cuotas = buscar_cuota_flexible(loc, vis, cuotas_reales)

        # Estructurar bloque de texto informativo
        lineas_reporte.append(f"PARTIDO: {loc} vs {vis} (Neutral: {'Sí' if neu==1 else 'No'})")
        lineas_reporte.append(f"  [Random Forest] Predicción Goles: {g_l_rf} - {g_v_rf}")
        lineas_reporte.append(f"  [SVR Machine]  Predicción Goles: {g_l_svr} - {g_v_svr}")
        lineas_reporte.append(f"  [Cuotas Odds API] Local: {cuotas['cuota_local']} | Empate: {cuotas['cuota_empate']} | Visitante: {cuotas['cuota_visitante']}")
        lineas_reporte.append("-" * 55)

    # Persistencia en repositorio
    with open("predicciones_reporte.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_reporte))
    print("-> ¡Proceso completado con éxito! Archivo 'predicciones_reporte.txt' guardado.")

if __name__ == "__main__":
    ejecutar_sistema_prediccion()
