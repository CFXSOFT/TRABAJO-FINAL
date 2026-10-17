# SENATI · Taller Regresión ML/DL — Casos 01 y 02

Una sola aplicación Flask con los dos casos prácticos intermedios (Python, Flask, HTML y CSS; sin JavaScript).
Un botón fijo abajo a la derecha cambia de un caso al otro.

| Caso | Ruta | Variables X | Respuesta y |
|------|------|-------------|-------------|
| 01 Pedidos de kits de robótica | `/caso-01` | consultas_web, inversion_publicidad_soles, campana_escolar | pedidos_kits |
| 02 Transporte agrícola | `/caso-02` | distancia_km, carga_kg, via_no_asfaltada | costo_transporte_soles |

## Estructura
```
app.py                       Flask: lee modelo.json, valida rangos, calcula la estimación
casos.py                     Configuración de los dos casos (campos, rangos, pruebas)
entrenar.py                  Entrena con Scikit-learn y escribe modelo.json de cada caso
templates/index.html         Página única para ambos casos
static/estilos.css           Estilos responsive (celular y PC)
caso_01_pedidos_kits/        datos.csv + modelo.json
caso_02_transporte_agricola/ datos.csv + modelo.json
notebooks/                   Cuadernos ejecutados de cada caso
public/static/               Copia de static/ para Vercel
```

## Resultados (20 % de prueba)
| Caso | MAE | R² |
|------|-----|----|
| 01 | 2.95 pedidos | 0.791 |
| 02 | S/ 17.79 | 0.977 |

Pruebas obligatorias: Caso 01 → 9.80 y 30.96 pedidos. Caso 02 → S/ 133.79 y S/ 454.13.

## Probar en tu computadora
```bash
pip install -r requirements.txt
python app.py
```
Abre http://127.0.0.1:5000. Para volver a entrenar: `pip install -r requirements-entrenamiento.txt` y luego `python entrenar.py`.

## Publicar gratis en Vercel
1. Sube esta carpeta a un repositorio de GitHub.
2. En vercel.com: **Add New → Project**, elige el repositorio y pulsa **Deploy** (detecta Flask solo).
3. Copia el link `https://tu-proyecto.vercel.app` y envíalo para la calificación.

Si cambias `static/`, copia también los archivos a `public/static/`.

Los datos son **simulados**; las estimaciones no son cotizaciones ni garantías.
