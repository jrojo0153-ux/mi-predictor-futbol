import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def ejecutar_sistema_prediccion():
    print("=== Iniciando el Sistema de Predicción de Fútbol ===")

    # 1. DATASET HISTÓRICO (Simulación de partidos acumulados)
    data_historica = {
        'home_team': ['Argentina', 'Brasil', 'Colombia', 'Chile', 'Uruguay', 'Perú', 'Ecuador', 'Argentina', 'Brasil', 'Colombia'],
        'away_team': ['Chile', 'Uruguay', 'Perú', 'Ecuador', 'Argentina', 'Brasil', 'Colombia', 'Brasil', 'Chile', 'Uruguay'],
        'home_score': [2, 1, 0, 3, 1, 0, 2, 1, 3, 2],
        'away_score': [1, 1, 1, 0, 2, 2, 1, 0, 1, 2],
        'neutral': [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data_historica)

    # 2. PROCESAMIENTO Y CODIFICACIÓN (Label Encoding)
    le = LabelEncoder()
    all_teams = list(set(df['home_team'].tolist() + df['away_team'].tolist()))
    le.fit(all_teams)
    
    df['home_encoded'] = le.transform(df['home_team'])
    df['away_encoded'] = le.transform(df['away_team'])

    # Variables de entrada y salida
    X = df[['home_encoded', 'away_encoded', 'neutral']]
    y_home = df['home_score']
    y_away = df['away_score']

    # 3. ENTRENAMIENTO DE MODELOS
    # Random Forest
    rf_home = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_home)
    rf_away = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_away)
    
    # SVR
    svr_home = SVR(kernel='rbf', C=1.0).fit(X, y_home)
    svr_away = SVR(kernel='rbf', C=1.0).fit(X, y_away)

    # 4. PARTIDOS PRÓXIMOS A PREDECIR (Fase de Torneo / Cartelera)
    partidos_nuevos = [
        {"local": "Argentina", "visitante": "Brasil", "neutral": 1},
        {"local": "Colombia", "visitante": "Uruguay", "neutral": 0},
        {"local": "Chile", "visitante": "Ecuador", "neutral": 0}
    ]

    lineas_resultado = ["=== RESULTADOS DE LAS PREDICCIONES ===\n"]

    for partido in partidos_nuevos:
        loc = partido["local"]
        vis = partido["visitante"]
        neu = partido["neutral"]
        
        # Formatear entrada numéricamente
        id_l = le.transform([loc])[0]
        id_v = le.transform([vis])[0]
        input_m = pd.DataFrame([[id_l, id_v, neu]], columns=['home_encoded', 'away_encoded', 'neutral'])
        
        # Predicción Random Forest
        goles_l_rf = int(np.round(rf_home.predict(input_m)[0]))
        goles_v_rf = int(np.round(rf_away.predict(input_m)[0]))
        
        # Predicción SVR
        goles_l_svr = int(np.round(svr_home.predict(input_m)[0]))
        goles_v_svr = int(np.round(svr_away.predict(input_m)[0]))
        
        # Guardar en el reporte
        lineas_resultado.append(f"Partido: {loc} vs {vis} (Neutral: {neu})")
        lineas_resultado.append(f"  -> Predicción Random Forest: {goles_l_rf} - {goles_v_rf}")
        lineas_resultado.append(f"  -> Predicción SVR: {goles_l_svr} - {goles_v_svr}\n")

    # 5. GUARDAR EL REPORTE EN UN ARCHIVO
    with open("predicciones_reporte.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_resultado))
        
    print("Reporte 'predicciones_reporte.txt' generado correctamente.")

if __name__ == "__main__":
    ejecutar_sistema_prediccion()
