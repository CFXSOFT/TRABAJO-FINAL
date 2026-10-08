"""
Sistema de apoyo para la detección temprana de riesgo de deserción estudiantil.
Aplicación Flask que carga el pipeline entrenado (modelo_desercion.pkl) y estima
el riesgo de abandono de un estudiante a partir de sus datos.

IMPORTANTE: es una herramienta de apoyo para tutores. No toma decisiones automáticas.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent

# Los archivos estáticos viven en public/ (Vercel sirve esa carpeta por CDN).
# Con static_url_path="" también funcionan en local: /css/estilos.css
app = Flask(__name__, static_folder=str(BASE_DIR / "public"), static_url_path="")

# ---------------------------------------------------------------------------
# Carga del modelo (se hace una sola vez al iniciar la aplicación)
# ---------------------------------------------------------------------------
MODELO = joblib.load(BASE_DIR / "modelo" / "modelo_desercion.pkl")
with open(BASE_DIR / "modelo" / "metricas.json", encoding="utf-8") as f:
    INFO = json.load(f)

COLUMNAS = list(MODELO.feature_names_in_)          # orden exacto que espera el pipeline
UMBRAL_BAJO = INFO["umbrales_riesgo"]["bajo_menor_a"]   # 0.25
UMBRAL_ALTO = INFO["umbrales_riesgo"]["alto_desde"]     # 0.50

# ---------------------------------------------------------------------------
# Definición del formulario (secciones, campos, rangos y ayudas)
# ---------------------------------------------------------------------------
SEDES = ["Arequipa", "Huancayo", "Independencia", "Lima Norte", "San Juan de Lurigancho", "Trujillo"]
PROGRAMAS = ["Desarrollo de Software", "Diseño y Desarrollo Web", "Ingeniería de Software con IA",
             "Redes y Seguridad Informática", "Soporte y Mantenimiento"]

SECCIONES = [
    {"titulo": "Perfil del estudiante", "icono": "user", "campos": [
        {"name": "sede", "label": "Sede", "tipo": "select", "opciones": SEDES},
        {"name": "programa", "label": "Programa de estudios", "tipo": "select", "opciones": PROGRAMAS},
        {"name": "turno", "label": "Turno", "tipo": "pills", "opciones": ["Mañana", "Tarde", "Noche"]},
        {"name": "semestre", "label": "Semestre", "tipo": "number", "min": 1, "max": 6, "step": 1},
        {"name": "edad", "label": "Edad", "tipo": "number", "min": 15, "max": 60, "step": 1},
    ]},
    {"titulo": "Rendimiento y asistencia", "icono": "chart", "campos": [
        {"name": "promedio_academico", "label": "Promedio académico", "tipo": "number", "min": 0, "max": 20,
         "step": 0.1, "opcional": True, "ayuda": "Escala de 0 a 20"},
        {"name": "asistencia_pct", "label": "Asistencia (%)", "tipo": "number", "min": 0, "max": 100,
         "step": 0.1, "opcional": True},
        {"name": "faltas_mes", "label": "Faltas al mes", "tipo": "number", "min": 0, "max": 30, "step": 1},
        {"name": "tardanzas_mes", "label": "Tardanzas al mes", "tipo": "number", "min": 0, "max": 30, "step": 1},
        {"name": "cursos_desaprobados", "label": "Cursos desaprobados", "tipo": "number", "min": 0, "max": 15,
         "step": 1},
    ]},
    {"titulo": "Hábitos y participación", "icono": "book", "campos": [
        {"name": "horas_estudio_semana", "label": "Horas de estudio por semana", "tipo": "number", "min": 0,
         "max": 80, "step": 0.1, "opcional": True},
        {"name": "uso_plataforma_horas", "label": "Uso de la plataforma (h/semana)", "tipo": "number", "min": 0,
         "max": 80, "step": 0.1, "opcional": True},
        {"name": "tutorias_mes", "label": "Tutorías al mes", "tipo": "number", "min": 0, "max": 20, "step": 1},
        {"name": "satisfaccion_formacion_1a5", "label": "Satisfacción con la formación", "tipo": "number",
         "min": 1, "max": 5, "step": 1, "opcional": True, "ayuda": "De 1 (baja) a 5 (alta)"},
    ]},
    {"titulo": "Contexto personal", "icono": "home", "campos": [
        {"name": "trabaja_actualmente", "label": "¿Trabaja actualmente?", "tipo": "pills", "opciones": ["No", "Sí"]},
        {"name": "horas_trabajo_semana", "label": "Horas de trabajo por semana", "tipo": "number", "min": 0,
         "max": 80, "step": 1},
        {"name": "distancia_centro_km", "label": "Distancia a la sede (km)", "tipo": "number", "min": 0,
         "max": 150, "step": 0.1},
        {"name": "tiempo_traslado_min", "label": "Tiempo de traslado (min)", "tipo": "number", "min": 0,
         "max": 300, "step": 1},
        {"name": "situacion_pago", "label": "Situación de pago", "tipo": "pills",
         "opciones": ["Al día", "1 cuota pendiente", "2 o más cuotas pendientes"]},
        {"name": "calidad_conectividad", "label": "Calidad de conectividad", "tipo": "pills",
         "opciones": ["Buena", "Regular", "Limitada"], "opcional": True},
        {"name": "apoyo_familiar", "label": "Apoyo familiar", "tipo": "pills",
         "opciones": ["Alto", "Medio", "Bajo"], "opcional": True},
    ]},
]
CAMPOS = {c["name"]: c for s in SECCIONES for c in s["campos"]}
assert set(CAMPOS) == set(COLUMNAS), "El formulario debe coincidir con las variables del modelo"

# Ejemplos precargados para probar la aplicación rápidamente
EJEMPLOS = {
    "alto": {"sede": "Arequipa", "programa": "Desarrollo de Software", "turno": "Tarde", "semestre": "3",
             "edad": "21", "promedio_academico": "10.5", "asistencia_pct": "68", "faltas_mes": "6",
             "tardanzas_mes": "4", "cursos_desaprobados": "3", "horas_estudio_semana": "4",
             "uso_plataforma_horas": "2.5", "tutorias_mes": "0", "satisfaccion_formacion_1a5": "2",
             "trabaja_actualmente": "Sí", "horas_trabajo_semana": "40", "distancia_centro_km": "18",
             "tiempo_traslado_min": "85", "situacion_pago": "2 o más cuotas pendientes",
             "calidad_conectividad": "Limitada", "apoyo_familiar": "Bajo"},
    "bajo": {"sede": "Independencia", "programa": "Ingeniería de Software con IA", "turno": "Mañana",
             "semestre": "4", "edad": "20", "promedio_academico": "16", "asistencia_pct": "95",
             "faltas_mes": "0", "tardanzas_mes": "1", "cursos_desaprobados": "0",
             "horas_estudio_semana": "13", "uso_plataforma_horas": "8", "tutorias_mes": "2",
             "satisfaccion_formacion_1a5": "5", "trabaja_actualmente": "No", "horas_trabajo_semana": "0",
             "distancia_centro_km": "5", "tiempo_traslado_min": "20", "situacion_pago": "Al día",
             "calidad_conectividad": "Buena", "apoyo_familiar": "Alto"},
}

ETIQUETAS = {
    "sede": "Sede", "programa": "Programa", "turno": "Turno", "semestre": "Semestre", "edad": "Edad",
    "promedio_academico": "Promedio académico", "asistencia_pct": "Asistencia",
    "faltas_mes": "Faltas al mes", "tardanzas_mes": "Tardanzas al mes",
    "cursos_desaprobados": "Cursos desaprobados", "horas_estudio_semana": "Horas de estudio",
    "uso_plataforma_horas": "Uso de la plataforma", "tutorias_mes": "Tutorías al mes",
    "satisfaccion_formacion_1a5": "Satisfacción con la formación",
    "trabaja_actualmente": "Situación laboral", "horas_trabajo_semana": "Horas de trabajo",
    "distancia_centro_km": "Distancia a la sede", "tiempo_traslado_min": "Tiempo de traslado",
    "situacion_pago": "Situación de pago", "calidad_conectividad": "Conectividad",
    "apoyo_familiar": "Apoyo familiar",
}

# Sugerencias de seguimiento cuando una variable aumenta el riesgo del estudiante
SUGERENCIAS = {
    "faltas_mes": "Conversar sobre las causas de las inasistencias y acordar un compromiso de asistencia.",
    "asistencia_pct": "Identificar barreras para asistir (horario, traslado, trabajo) y buscar alternativas.",
    "promedio_academico": "Ofrecer reforzamiento académico en los cursos con menor nota.",
    "cursos_desaprobados": "Evaluar un plan de recuperación de los cursos desaprobados.",
    "apoyo_familiar": "Explorar redes de apoyo y derivar a bienestar estudiantil u orientación.",
    "situacion_pago": "Derivar a consejería económica para revisar facilidades de pago.",
    "calidad_conectividad": "Verificar el acceso a internet y ofrecer alternativas dentro de la sede.",
    "horas_trabajo_semana": "Coordinar horarios y evaluar flexibilidad con el turno de estudio.",
    "tiempo_traslado_min": "Considerar un turno u opciones que reduzcan el tiempo de traslado.",
    "distancia_centro_km": "Considerar un turno u opciones que reduzcan el tiempo de traslado.",
    "tutorias_mes": "Incentivar la participación en tutorías.",
    "horas_estudio_semana": "Orientar sobre la organización del tiempo de estudio.",
    "uso_plataforma_horas": "Motivar el uso de la plataforma académica.",
    "satisfaccion_formacion_1a5": "Conversar sobre su experiencia y expectativas con la formación.",
}

NIVELES = {
    "Bajo": {
        "clase": "bajo", "titulo": "Riesgo bajo",
        "interpretacion": "El perfil del estudiante se parece al de quienes continúan sus estudios. "
                          "No se observan señales fuertes de abandono.",
        "recomendacion": "Mantener el seguimiento académico regular y reforzar los hábitos positivos.",
    },
    "Medio": {
        "clase": "medio", "titulo": "Riesgo medio",
        "interpretacion": "El perfil presenta algunas señales que conviene vigilar. Todavía hay "
                          "buen margen para intervenir de forma preventiva.",
        "recomendacion": "Programar una conversación preventiva con el tutor en las próximas semanas "
                         "y revisar los factores principales.",
    },
    "Alto": {
        "clase": "alto", "titulo": "Riesgo alto",
        "interpretacion": "El perfil es muy parecido al de estudiantes que abandonaron. "
                          "Se recomienda atención prioritaria.",
        "recomendacion": "Realizar una revisión prioritaria: reunión individual con el tutor, "
                         "análisis de la situación personal y un plan de apoyo con seguimiento.",
    },
}


def clasificar_riesgo(prob: float) -> str:
    """Convierte la probabilidad en un nivel de riesgo (decisión de diseño documentada)."""
    if prob < UMBRAL_BAJO:
        return "Bajo"
    if prob < UMBRAL_ALTO:
        return "Medio"
    return "Alto"


def _variable_original(nombre_salida: str, columnas_cat: list) -> str:
    """Relaciona una columna transformada (p. ej. 'cat__sede_Lima Norte') con su variable original."""
    if nombre_salida.startswith("num__"):
        return nombre_salida[5:]
    sin_prefijo = nombre_salida[5:]
    for col in sorted(columnas_cat, key=len, reverse=True):
        if sin_prefijo.startswith(col + "_"):
            return col
    return sin_prefijo


def explicar(fila: pd.DataFrame) -> list:
    """Aporte de cada variable al puntaje del estudiante (coeficiente x valor estandarizado)."""
    prep = MODELO.named_steps["prep"]
    clf = MODELO.named_steps["modelo"]
    z = prep.transform(fila)
    z = z.toarray()[0] if hasattr(z, "toarray") else np.asarray(z)[0]
    aportes = z * clf.coef_[0]
    cols_cat = INFO["variables_categoricas"]
    por_variable = {}
    for nombre, valor in zip(prep.get_feature_names_out(), aportes):
        var = _variable_original(nombre, cols_cat)
        por_variable[var] = por_variable.get(var, 0.0) + float(valor)
    return sorted(por_variable.items(), key=lambda kv: kv[1], reverse=True)


def validar(form) -> tuple:
    """Valida el formulario. Devuelve (datos_para_el_modelo, errores)."""
    datos, errores = {}, {}
    for nombre, campo in CAMPOS.items():
        bruto = (form.get(nombre) or "").strip()
        if bruto == "":
            if campo.get("opcional"):
                datos[nombre] = np.nan          # el pipeline lo imputará (mediana o moda)
            else:
                errores[nombre] = "Este dato es obligatorio."
            continue
        if campo["tipo"] in ("select", "pills"):
            if bruto not in campo["opciones"]:
                errores[nombre] = "Selecciona una opción válida."
            else:
                datos[nombre] = bruto
        else:
            try:
                valor = float(bruto.replace(",", "."))
            except ValueError:
                errores[nombre] = "Ingresa un número válido."
                continue
            if not (campo["min"] <= valor <= campo["max"]):
                errores[nombre] = f"Debe estar entre {campo['min']} y {campo['max']}."
            else:
                datos[nombre] = valor
    return datos, errores


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", info=INFO)


@app.route("/predecir", methods=["GET", "POST"])
def predecir():
    if request.method == "GET":
        valores = EJEMPLOS.get(request.args.get("ejemplo"), {})
        return render_template("predecir.html", secciones=SECCIONES, valores=valores, errores={})

    datos, errores = validar(request.form)
    if errores:
        return render_template("predecir.html", secciones=SECCIONES, valores=request.form,
                               errores=errores), 400

    fila = pd.DataFrame([datos], columns=COLUMNAS)
    probabilidad = float(MODELO.predict_proba(fila)[0, 1])
    nivel = clasificar_riesgo(probabilidad)

    aportes = explicar(fila)
    max_abs = max(abs(v) for _, v in aportes) or 1.0
    suben = [(k, v) for k, v in aportes if v > 0.03][:4]
    bajan = [(k, v) for k, v in reversed(aportes) if v < -0.03][:3]
    factores_suben = [{"nombre": ETIQUETAS[k], "ancho": round(abs(v) / max_abs * 100)} for k, v in suben]
    factores_bajan = [{"nombre": ETIQUETAS[k], "ancho": round(abs(v) / max_abs * 100)} for k, v in bajan]
    sugerencias = []
    for k, _ in suben:
        texto = SUGERENCIAS.get(k)
        if texto and texto not in sugerencias:
            sugerencias.append(texto)

    return render_template(
        "resultado.html",
        porcentaje=round(probabilidad * 100),
        porcentaje_exacto=round(probabilidad * 100, 1),
        nivel=nivel, datos_nivel=NIVELES[nivel],
        factores_suben=factores_suben, factores_bajan=factores_bajan,
        sugerencias=sugerencias[:3], info=INFO,
    )


@app.route("/modelo")
def modelo():
    etiquetas_vars = [ETIQUETAS[v] for v in INFO["variables_numericas"] + INFO["variables_categoricas"]]
    return render_template("modelo.html", info=INFO, variables=etiquetas_vars,
                           suben=[ETIQUETAS.get(v, v) for v in _nombres_variables("variables_que_aumentan_riesgo")],
                           bajan=[ETIQUETAS.get(v, v) for v in _nombres_variables("variables_que_reducen_riesgo")])


def _nombres_variables(clave: str) -> list:
    """Convierte nombres como 'apoyo_familiar_Bajo' en la variable original (sin duplicados)."""
    vistos = []
    for nombre in INFO[clave]:
        var = _variable_original("cat__" + nombre, INFO["variables_categoricas"]) \
            if nombre not in INFO["variables_numericas"] else nombre
        if var not in vistos:
            vistos.append(var)
    return vistos


@app.errorhandler(404)
def no_encontrado(_):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
