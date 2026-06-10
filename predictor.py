import os
import requests
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# =========================================================================
# CONFIGURACIÓN COMPLETA - INFRAESTRUCTURA MLOps (BGLogicSolutions)
# =========================================================================
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
SPORT_KEY = "soccer"  # Forzado a fútbol global para evitar mezclas con la MLB
REGIONS = "us,eu"                    
MARKETS = "h2h"                      
CSV_HISTORICO = "dataset_historico.csv"
CSV_METRICAS = "metricas_rendimiento.csv"

def inicializar_o_cargar_historico():
    """Carga el dataset evolutivo o lo inicializa con los datos semilla"""
    if os.path.exists(CSV_HISTORICO):
        return pd.read_csv(CSV_HISTORICO)
    else:
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
    """Consulta ESPN para buscar partidos finalizados y sumarlos a la base de conocimiento"""
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.friendly/scoreboard"
    nuevas_filas = []
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for event in data.get('events', []):
                status = event.get('status', {}).get('type', {}).get('completed', False)
                if not status: continue  # Solo partidos terminados
                
                competitions = event.get('competitions', [])
                if not competitions: continue
                competitors = competitions[0].get('competitors', [])
                
                try:
                    home_data = next(c for c in competitors if c.get('homeAway') == 'home')
                    away_data = next(c for c in competitors if c.get('homeAway') == 'away')
                    
                    home_name = home_data['team']['displayName']
                    away_name = away_data['team']['displayName']
                    home_score = int(home_data['score'])
                    away_score = int(away_data['score'])
                    neutral = 1 if competitions[0].get('neutralGames', False) else 0
                    
                    # Evitar duplicar registros ya existentes
                    duplicado = df_historico[
                        (df_historico['home_team'] == home_name) & 
                        (df_historico['away_team'] == away_name) & 
                        (df_historico['home_score'] == home_score) &
                        (df_historico['away_score'] == away_score)
                    ]
                    
                    if duplicado.empty:
                        print(f"[NUEVO DATO ENCONTRADO]: {home_name} {home_score} - {away_score} {away_name}")
                        nuevas_filas.append({
                            'home_team': home_name, 'away_team': away_name,
                            'home_score': home_score, 'away_score': away_score,
                            'neutral': neutral
                        })
                except Exception: continue
                    
            if nuevas_filas:
                df_nuevos_datos = pd.DataFrame(nuevas_filas)
                df_actualizado = pd.concat([df_historico, df_nuevos_datos], ignore_index=True)
                df_actualizado.to_csv(CSV_HISTORICO, index=False)
                return df_actualizado
        return df_historico
    except Exception as e:
        print(f"[MLOps Error] No se pudo actualizar el histórico: {e}")
        return df_historico

def registrar_metricas_auditoria(df, mae_h, rmse_h, mae_a, rmse_a):
    """Guarda en un archivo log de control el comportamiento de los errores del modelo"""
    fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nuevas_metricas = {
        "fecha": [fecha_actual],
        "partidos_totales": [len(df)],
        "mae_goles_local": [mae_h],
        "rmse_goles_local": [rmse_h],
        "mae_goles_vis": [mae_a],
        "rmse_goles_vis": [rmse_a]
    }
    df_nuevas = pd.DataFrame(nuevas_metricas)
    if os.path.exists(CSV_METRICAS):
        df_log = pd.read_csv(CSV_METRICAS)
        df_log = pd.concat([df_log, df_nuevas], ignore_index=True)
    else:
        df_log = df_nuevas
    df_log.to_csv(CSV_METRICAS, index=False)

def obtener_partidos_espn():
    """Extrae partidos agendados (futuros) desde el scoreboard de ESPN"""
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.friendly/scoreboard"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        partidos = []
        for event in data.get('events', []):
            status = event.get('status', {}).get('type', {}).get('completed', False)
            if status: continue  # Saltar partidos cerrados
            
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
    except Exception: return []

def obtener_cuotas_the_odds():
    """Consulta las líneas de dinero activas de fútbol global"""
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
    print("=== [BGLogicSolutions] Iniciando Pipeline con Aprendizaje y Métricas ===")

    # 1. ACTUALIZAR HISTÓRICO RECOLECTANDO RESULTADOS REALES RECIENTES
    df = inicializar_o_cargar_historico()
    df = actualizar_historico_con_resultados_reales(df)

    partidos_nuevos = obtener_partidos_espn()
    cuotas_reales = obtener_cuotas_the_odds()

    if not partidos_nuevos:
        print("-> Sin partidos en agenda ESPN hoy. Usando cartelera de simulación.")
        partidos_nuevos = [
            {"local": "México", "visitante": "Estados Unidos", "neutral": 1},
            {"local": "Argentina", "visitante": "Brasil", "neutral": 1},
            {"local": "Francia", "visitante": "Alemania", "neutral": 0}
        ]

    # 2. ENCODER DE VARIABLES CATEGÓRICAS
    le = LabelEncoder()
    todos_los_equipos = list(set(
        df['home_team'].tolist() + df['away_team'].tolist() + 
        [p['local'] for p in partidos_nuevos] + [p['visitante'] for p in partidos_nuevos]
    ))
    le.fit(todos_los_equipos)
    df['home_encoded'] = le.transform(df['home_team'])
    df['away_encoded'] = le.transform(df['away_team'])

    # 3. CONSTRUCCIÓN DE MATRICES Y FIT DE ALGORITMOS
    X = df[['home_encoded', 'away_encoded', 'neutral']]
    
    rf_home = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['home_score'])
    rf_away = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['away_score'])
    svr_home = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, df['home_score'])
    svr_away = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, df['away_score'])

    # 4. SISTEMA DE AUTOCONTROL Y MEDICIÓN DE MÉTRICAS (Loss Evaluation)
    pred_rf_h = rf_home.predict(X)
    pred_rf_a = rf_away.predict(X)
    
    mae_h = mean_absolute_error(df['home_score'], pred_rf_h)
    rmse_h = root_mean_squared_error(df['home_score'], pred_rf_h)
    mae_a = mean_absolute_error(df['away_score'], pred_rf_a)
    rmse_a = root_mean_squared_error(df['away_score'], pred_rf_a)

    # Guardar las métricas de error calculadas en esta iteración para auditar la evolución
    registrar_metricas_auditoria(df, mae_h, rmse_h, mae_a, rmse_a)
    print(f"-> Métricas calculadas sobre histórico de {len(df)} filas. Log de rendimiento actualizado.")

    # 5. GENERACIÓN DEL REPORTE DE PREDICCIONES (Se mantiene guardándose en tu archivo TXT)
    lineas_reporte = [
        "==========================================================",
        " REPORTE CONSOLIDADO: AMISTOSOS INTERNACIONALES (IA)     ",
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
    print("-> Archivo 'predicciones_reporte.txt' y registros MLOps guardados con éxito.")

if __name__ == "__main__":
    ejecutar_sistema_prediccion()
