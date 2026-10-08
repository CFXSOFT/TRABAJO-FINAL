# Sistema de detección de riesgo de deserción estudiantil

Trabajo final · **Fundamentos y Algoritmia para Inteligencia Artificial** · Instructor: Ing. Gunar Castro Iñigo

Aplicación web (Python + Flask + HTML + CSS) que estima el riesgo de abandono de un estudiante con un modelo de
**Regresión Logística**. Es una herramienta de **apoyo para tutores**: no toma decisiones automáticas.

> El dataset es **sintético con comportamiento realista**. No contiene información real de estudiantes.

## Resultados del modelo (120 estudiantes de prueba)

| Recall (abandono) | Precision | F1 | Accuracy | ROC-AUC |
|---:|---:|---:|---:|---:|
| 0.65 | 0.37 | 0.47 | 0.76 | 0.74 |

Detecta 13 de los 20 estudiantes que abandonan. Se priorizó el recall porque un falso negativo es el error más costoso.

## Estructura del proyecto

```text
TRABAJO_FINAL_DESERCION/
├── 01_JUPYTER/analisis_desercion.ipynb      # Etapas 1 a 8 (ya ejecutado)
├── 02_DATASET/dataset_riesgo_desercion_estudiantil.csv
├── 03_MODELO/                               # modelo_desercion.pkl + metricas.json
├── 04_APLICATIVO/                           # aplicación Flask (esta carpeta se despliega en Vercel)
│   ├── app.py
│   ├── requirements.txt
│   ├── vercel.json
│   ├── modelo/                              # copia del .pkl y metricas.json que usa la app
│   ├── templates/                           # index, predecir, resultado, modelo, 404
│   └── public/                              # CSS, icono y manifest (archivos estáticos)
├── 05_DOCUMENTACION/                        # informe_final.pdf, gráficos y capturas
└── README.md
```

> **Nota sobre `public/` en lugar de `static/`:** Vercel sirve los archivos estáticos de Flask desde la carpeta
> `public/`. Flask está configurado para usar esa misma carpeta, así que la app funciona igual en local y en Vercel.

## Ejecutar en local

```bash
cd 04_APLICATIVO
python -m venv .venv
# Windows: .venv\Scripts\activate      |  Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Abrir <http://127.0.0.1:5000>. Para probar desde el celular en la misma red Wi-Fi, ejecutar con
`flask run --host 0.0.0.0` y abrir `http://IP-DE-LA-PC:5000`.

## Volver a entrenar el modelo

Abrir `01_JUPYTER/analisis_desercion.ipynb` y ejecutar todas las celdas. El cuaderno regenera
`modelo_desercion.pkl` y `metricas.json` en `03_MODELO/` y en `04_APLICATIVO/modelo/`.

> El `.pkl` solo es compatible con la **misma versión de scikit-learn** con la que se entrenó (1.8.0).
> Por eso `requirements.txt` fija las versiones exactas. Si se cambian, hay que reentrenar.

## Publicar en GitHub y Vercel

1. **GitHub:** crear un repositorio nuevo y subir toda la carpeta `TRABAJO_FINAL_DESERCION`.
   ```bash
   git init
   git add .
   git commit -m "Trabajo final: riesgo de deserción estudiantil"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
   git push -u origin main
   ```
2. **Vercel:** entrar a vercel.com, elegir *Add New → Project* e importar el repositorio de GitHub.
3. En la configuración del proyecto, poner **Root Directory = `04_APLICATIVO`**. Vercel detecta Flask solo
   (busca `app.py` con la variable `app`), instala `requirements.txt` y publica.
4. Pulsar **Deploy**. Cada `git push` posterior vuelve a desplegar automáticamente.

Si el despliegue falla por **tamaño de la función** (scikit-learn, SciPy y Pandas son pesados), avisar para
preparar una versión ligera de la app.

## Cómo funciona la predicción

1. `app.py` carga una sola vez `modelo/modelo_desercion.pkl` (pipeline completo: imputación, escalado, codificación y regresión logística).
2. El formulario envía los 21 datos del estudiante; el servidor los valida (tipos y rangos).
3. Se arma un `DataFrame` de una fila y se llama a `predict_proba`.
4. La probabilidad se convierte en nivel: **bajo** (< 25 %), **medio** (25 % a < 50 %) o **alto** (≥ 50 %).
5. Se muestran los factores que más pesan en ese caso (coeficiente × valor estandarizado) y acciones de seguimiento sugeridas.

## Limitaciones

Datos sintéticos, muestra pequeña (20 casos de abandono en la prueba), ROC-AUC moderado y variables que no
capturan motivos personales o de salud. Como el modelo usa pesos de clase balanceados, el valor mostrado es un
índice de riesgo y no una probabilidad real calibrada. Es un apoyo a la revisión académica, nunca una decisión definitiva.
