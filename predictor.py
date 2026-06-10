import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# =========================================================================
# CONFIGURACIÓN COMPLETA - PIPELINE AUTÓNOMO CON AUTOENTRENAMIENTO
# =========================================================================
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
SPORT_KEY = "soccer"  
REGIONS = "us,eu"                    
MARKETS = "h2h"                      
CSV_HISTORICO = "dataset_historico.csv"

def inicializar_o_cargar_historico():
    """Carga el dataset evolutivo o lo crea desde cero si es la primera corrida"""
    if os.path.exists(CSV_HISTORICO):
        print(f"-> Cargando dataset evolutivo desde {CSV_HISTORICO}...")
        return pd.read_csv(CSV_HISTORICO)
    else:
        print("-> No se encontró histórico previo. Inicializando base de datos semilla...")
        data_semilla = {
            'home_team': ['México', 'Estados Unidos', 'Argentina', 'Brasil', 'Colombia', 'España', 'Alemania', 'México', 'Francia', 'Argentina'],
            'away_team': ['Estados Unidos', 'Colombia', 'Brasil', 'México', 'Alemania', 'Brasil', 'Francia', 'Argentina', 'Chile', 'Ecuador'],
            'home_score': [1, 1, 2, 0, 3, 0, 1, 2, 3, 1],
            'away_score': [2, 5, 0, 1, 2, 1, 2, 0, 2, 0],
            'neutral': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]
        }
        df = pd.DataFrame(data_semilla)
        df.to_csv(CSV_HISTORICO, index=False)
        return df

def actualizar_historico_con_resultados_reales(df_historico):
    """Consulta ESPN para buscar partidos finalizados de selecciones y los añade al dataset"""
    print("-> Buscando resultados recientes en ESPN para autoentrenamiento...")
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.friendly/scoreboard"
    nuevas_filas = []
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for event in data.get('events', []):
                # Validar si el partido YA TERMINÓ
                status = event.get('status', {}).get('type', {}).get('completed', False)
                if not status:
                    continue # Saltarse partidos en vivo o futuros
                
                competitions = event.get('competitions', [])
                if not competitions: continue
                competitors = competitions[0].get('competitors', [])
                
                try:
                    home_data = next(c for c in competitors if c.get('homeAway') == 'home')
                    away_data = next(c = next(c for c in competitors if c.get('homeAway') == 'away'))
                    
                    home_name = home_data['team']['displayName']
                    away_name = away_data['team']['displayName']
                    home_score = int(home_data['score'])
                    away_score = int(away_data['score'])
                    neutral = 1 if competitions[0].get('neutralGames', False) else 0
                    
                    # Evitar duplicados: verificar si este partido exacto con este marcador ya existe
                    duplicado = df_historico[
                        (df_historico['home_team'] == home_name) & 
                        (df_historico['away_team'] == away_name) & 
                        (df_historico['home_score'] == home_score) &
                        (df_historico['away_score'] == away_score)
                    ]
                    
                    if duplicado.empty:
                        print(f"   [Nuevo Dato Real Encontrado]: {home_name} {home_score} - {away_score} {away_name}")
                        nuevas_filas.append({
                            'home_team': home_name, 'away_team': away_name,
                            'home_score': home_score, 'away_score': away_score,
                            'neutral': neutral
                        })
                except Exception:
                    continue
                    
            if nuevas_filas:
                df_nuevos_datos = pd.DataFrame(nuevas_filas)
                df_actualizado = pd.concat([df_historico, df_nuevos_datos], ignore_index=True)
                df_actualizado.to_csv(CSV_HISTORICO, index=False)
                print(f"-> ¡Autoentrenamiento exitoso! Se añadieron {len(nuevas_filas)} nuevos ejemplos al histórico.")
                return df_actualizado
            else:
                print("-> No se detectaron partidos finalizados nuevos que no estuvieran ya guardados.")
        return df_historico
    except Exception as e:
        print(f"-> Saltando actualización de histórico por error de red: {e}")
        return df_historico

def obtener_partidos_espn():
    """Extrae la cartelera de los próximos partidos amistosos"""
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.friendly/scoreboard"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        partidos = []
        for event in data.get('events', []):
            # Solo predecir partidos que NO han terminado
            status = event.get('status', {}).get('type', {}).get('completed', False)
            if status: continue
            
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
    except Exception:
        return []

def obtener_cuotas_the_odds():
    """Extrae las cuotas vigentes globales de fútbol"""
    if not ODDS_API_KEY: return {}
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/"
    params = {'apiKey': ODDS_API_KEY, 'regions': REGIONS, 'markets': MARKETS, 'oddsFormat': 'decimal'}
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
                        if name == home_team: odds_dict['cuota_local'] = price
                        elif name == away_team: odds_dict['cuota_visitante'] = price
                        elif name.lower() in ['draw', 'empate']: odds_dict['cuota_empate'] = price
                    cuotas_mapeadas[f"{home_team.lower()}_{away_team.lower()}"] = odds_dict
            return cuotas_mapeadas
        return {}
    except Exception: return {}

def buscar_cuota_flexible(local, visitante, diccionario_cuotas):
    key_directa = f"{local.lower()}_{visitante.lower()}"
    if key_directa in diccionario_cuotas: return diccionario_cuotas[key_directa]
    for key, cuotas in diccionario_cuotas.items():
        api_local, api_visitante = key.split('_')
        if (local.lower() in api_local or api_local in local.lower()) or \
           (visitante.lower() in api_visitante or api_visitante in visitante.lower()):
            return cuotas
    return {"cuota_local": "N/D", "cuota_empate": "N/D", "cuota_visitante": "N/D"}

def ejecutar_sistema_prediccion():
    print("=== [BGLogicSolutions] Pipeline MLOps - Ciclo Continuo ===")

    # 1. CARGAR HISTÓRICO Y AUTOENTRENAR CON RESULTADOS REALES DE JUEGOS PASADOS
    df = inicializar_o_cargar_historico()
    df = actualizar_historico_con_resultados_reales(df)

    # 2. CAPTURAR PRÓXIMOS ENCUENTROS Y CUOTAS
    partidos_nuevos = obtener_partidos_espn()
    cuotas_reales = obtener_cuotas_the_odds()

    if not partidos_nuevos:
        print("-> Sin partidos nuevos en agenda ESPN hoy. Generando reporte simulado.")
        partidos_nuevos = [
            {"local": "México", "visitante": "Estados Unidos", "neutral": 1},
            {"local": "Argentina", "visitante": "Brasil", "neutral": 1}
        ]

    # 3. FIT ENCODER INTEGRANDO TODOS LOS EQUIPOS NUEVOS Y PASADOS
    le = LabelEncoder()
    todos_los_equipos = list(set(
        df['home_team'].tolist() + df['away_team'].tolist() + 
        [p['local'] for p in partidos_nuevos] + [p['visitante'] for p in partidos_nuevos]
    ))
    le.fit(todos_los_equipos)
    
    df['home_encoded'] = le.transform(df['home_team'])
    df['away_encoded'] = le.transform(df['away_team'])

    # 4. RE-ENTRENAMIENTO DINÁMICO DE LOS MODELOS REPLICADOS
    X = df[['home_encoded', 'away_encoded', 'neutral']]
    
    rf_home = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['home_score'])
    rf_away = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['away_score'])
    svr_home = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, df['home_score'])
    svr_away = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, df['away_score'])

    # 5. GENERACIÓN DEL REPORTE
    lineas_reporte = [
        "==========================================================",
        " REPORTE MLOps: MODELOS IA EVOLUTIVOS (MUNDIAL 2026)      ",
        f" Volumen del Dataset de Entrenamiento: {len(df)} partidos ",
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

        cuotas = buscar_cuota_flexible(loc, vis, cuotas_reales)

        lineas_reporte.append(f"PARTIDO: {loc} vs {vis} (Neutral: {'Sí' if neu==1 else 'No'})")
        lineas_reporte.append(f"  [Random Forest] Predicción Goles: {g_l_rf} - {g_v_rf}")
        lineas_reporte.append(f"  [SVR Machine]  Predicción Goles: {g_l_svr} - {g_v_svr}")
        lineas_reporte.append(f"  [Cuotas Odds API] Local: {cuotas['cuota_local']} | Empate: {cuotas['cuota_empate']} | Visitante: {cuotas['cuota_visitante']}")
        lineas_reporte.append("-" * 55)

    with open("predicciones_reporte.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_reporte))
    print("-> ¡Pipeline finalizado! Dataset e Inferencia actualizados.")

if __name__ == "__main__":
    ejecutar_sistema_prediccion()
