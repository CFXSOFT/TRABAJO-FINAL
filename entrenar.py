# entrenar.py | Entrenamiento de los dos casos con Scikit-learn.
# Se ejecuta UNA vez en la computadora:   python entrenar.py
# Lee datos.csv de cada caso, entrena LinearRegression, evalúa con el 20 % de prueba
# y guarda los coeficientes en modelo.json. La web (app.py) solo lee ese JSON,
# por eso en el hosting no hace falta instalar scikit-learn.

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from casos import CASOS

RAIZ = Path(__file__).resolve().parent


def entrenar_caso(clave, caso):
    carpeta = RAIZ / caso["carpeta"]

    # 1) Leer el CSV (utf-8-sig quita la marca BOM que trae el archivo).
    datos = pd.read_csv(carpeta / "datos.csv", encoding="utf-8-sig")

    # 2) Limpieza básica: ausentes y duplicados.
    ausentes = int(datos.isna().sum().sum())
    duplicados = int(datos.duplicated().sum())
    datos = datos.dropna().drop_duplicates()

    # 3) Variables de entrada X y respuesta y.
    columnas = [c["nombre"] for c in caso["campos"]]
    X = datos[columnas]
    y = datos[caso["objetivo"]]

    # 4) Separar 80 % entrenamiento / 20 % prueba (random_state fijo = resultado repetible).
    X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5) Entrenar la regresión lineal múltiple.
    modelo = LinearRegression()
    modelo.fit(X_ent, y_ent)

    # 6) Evaluar SOLO con los datos de prueba (que el modelo nunca vio).
    pred = modelo.predict(X_prueba)
    mae = mean_absolute_error(y_prueba, pred)
    r2 = r2_score(y_prueba, pred)

    # 7) Guardar todo lo que la web necesita.
    resumen = {
        "caso": clave,
        "columnas": columnas,
        "objetivo": caso["objetivo"],
        "intercepto": float(modelo.intercept_),
        "coeficientes": {c: float(k) for c, k in zip(columnas, modelo.coef_)},
        "mae": float(mae),
        "r2": float(r2),
        "n_total": int(len(datos)),
        "n_entrenamiento": int(len(X_ent)),
        "n_prueba": int(len(X_prueba)),
        "ausentes": ausentes,
        "duplicados": duplicados,
        "media_objetivo": float(y.mean()),
        "ejemplos_prueba": [
            {**{c: float(fila[c]) for c in columnas},
             "real": float(real), "predicho": float(p)}
            for (_, fila), real, p in list(zip(X_prueba.iterrows(), y_prueba, pred))[:8]
        ],
    }
    with open(carpeta / "modelo.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"{clave}: filas={len(datos)}  MAE={mae:.3f}  R2={r2:.3f}")
    print("  intercepto:", round(modelo.intercept_, 4))
    for c, k in zip(columnas, modelo.coef_):
        print(f"  {c}: {k:.4f}")


if __name__ == "__main__":
    for clave, caso in CASOS.items():
        entrenar_caso(clave, caso)
