import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def ejecutar_sistema_prediccion():
    print("=== Iniciando Pipeline Autónomo de Predicción ===")

    # 1. DATASET HISTÓRICO (Partidos de selecciones latinoamericanas)
    # Nota: Si usas un CSV externo, puedes cargarlo aquí con: pd.read_csv('tus_datos.csv')
    data_historica = {
        'home_team': ['Argentina', 'Brasil', 'Colombia', 'Chile', 'Uruguay', 'Perú', 'Ecuador', 'Argentina', 'Brasil', 'Colombia', 'Uruguay', 'Ecuador'],
        'away_team': ['Chile', 'Uruguay', 'Perú', 'Ecuador', 'Argentina', 'Brasil', 'Colombia', 'Brasil', 'Chile', 'Uruguay', 'Colombia', 'Argentina'],
        'home_score': [2, 1, 0, 3, 1, 0, 2, 1, 3, 2, 1, 0],
        'away_score': [1, 1, 1, 0, 2, 2, 1, 0, 1, 2, 0, 2],
        'neutral': [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data_historica)

    # 2. PROCESAMIENTO Y CODIFICACIÓN NUMÉRICA (Label Encoding)
    le = LabelEncoder()
    all_teams = list(set(df['home_team'].tolist() + df['away_team'].tolist()))
    le.fit(all_teams)
    
    df['home_encoded'] = le.transform(df['home_team'])
    df['away_encoded'] = le.transform(df['away_team'])

    # Variables de entrenamiento (X: características de entrada, y: objetivos de goles)
    X = df[['home_encoded', 'away_encoded', 'neutral']]
    y_home = df['home_score']
    y_away = df['away_score']

    # 3. ENTRENAMIENTO DE LOS MODELOS DE REGRESIÓN (Mismo enfoque del video de la UIS)
    # Algoritmo A: Random Forest Regressor
    rf_home = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_home)
    rf_away = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_away)
    
    # Algoritmo B: Support Vector Regressor (SVR)
    svr_home = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, y_home)
    svr_away = SVR(kernel='rbf', C=1.0, epsilon=0.2).fit(X, y_away)

    # 4. CARTELERA DE PARTIDOS A PREDECIR (Fase de Torneo a evaluar)
    partidos_nuevos = [
        {"local": "Argentina", "visitante": "Brasil", "neutral": 1},
        {"local": "Colombia", "visitante": "Uruguay", "neutral": 0},
        {"local": "Chile", "visitante": "Ecuador", "neutral": 0},
        {"local": "Uruguay", "visitante": "Argentina", "neutral": 0}
    ]

    lineas_resultado = [
        "==================================================",
        "  REPORTE AUTÓNOMO DE PREDICCIONES GENERADAS      ",
        "==================================================\n"
    ]

    # 5. EJECUCIÓN DE PREDICCIONES
    for partido in partidos_nuevos:
        loc = partido["local"]
        vis = partido["visitante"]
        neu = partido["neutral"]
        
        # Validar si los equipos ingresados se encuentran en el histórico codificado
        if loc in le.classes_ and vis in le.classes_:
            id_l = le.transform([loc])[0]
            id_v = le.transform([vis])[0]
            
            input_m = pd.DataFrame([[id_l, id_v, neu]], columns=['home_encoded', 'away_encoded', 'neutral'])
            
            # Predicciones de goles continuos -> Redondeo al entero más cercano (Fútbol Real)
            pred_l_rf = int(np.round(rf_home.predict(input_m)[0]))
            pred_v_rf = int(np.round(rf_away.predict(input_m)[0]))
            
            pred_l_svr = int(np.round(svr_home.predict(input_m)[0]))
            pred_v_svr = int(np.round(svr_away.predict(input_m)[0]))
            
            # Almacenar datos formateados
            lineas_resultado.append(f"Partido: {loc} vs {vis} (Campo Neutral: {'Sí' if neu==1 else 'No'})")
            lineas_resultado.append(f"  [Random Forest] Marcador Estimado: {pred_l_rf} - {pred_v_rf}")
            lineas_resultado.append(f"  [SVR Machine]  Marcador Estimado: {pred_l_svr} - {pred_v_svr}\n")
        else:
            lineas_resultado.append(f"Saltando partido: {loc} vs {vis} (Uno de los equipos no está en la base de datos)\n")

    # 6. EXPORTAR REPORTE AUTOMÁTICO
    output_filename = "predicciones_reporte.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_resultado))
        
    print(f"-> Proceso finalizado. Archivo '{output_filename}' actualizado.")

if __name__ == "__main__":
    ejecutar_sistema_prediccion()
