# app.py | Taller Regresión ML/DL - SENATI
# Una sola aplicación Flask con los dos casos prácticos intermedios:
#   /caso-01  Estimación de pedidos de kits de robótica
#   /caso-02  Estimación del costo de transporte agrícola
# Solo Python, Flask, HTML y CSS (sin JavaScript).
# Los coeficientes se leen de modelo.json (creado por entrenar.py), por eso
# en el hosting solo se necesita Flask.

import json
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, send_from_directory, url_for

from casos import CASOS

app = Flask(__name__)
RAIZ = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. Cargar los modelos guardados (modelo.json de cada caso)
# ---------------------------------------------------------------------------
def cargar_modelo(caso):
    with open(RAIZ / caso["carpeta"] / "modelo.json", encoding="utf-8") as f:
        return json.load(f)


MODELOS = {clave: cargar_modelo(caso) for clave, caso in CASOS.items()}


# ---------------------------------------------------------------------------
# 2. Predicción: intercepto + suma(coeficiente * valor)
# ---------------------------------------------------------------------------
def predecir(clave, valores):
    m = MODELOS[clave]
    total = m["intercepto"]
    for columna in m["columnas"]:
        total += m["coeficientes"][columna] * valores[columna]
    return max(total, 0)  # un pedido o un costo no pueden ser negativos


def formatear(clave, valor):
    if clave == "caso-01":
        return f"{valor:.0f} pedidos"
    return f"S/ {valor:,.2f}"


# ---------------------------------------------------------------------------
# 3. Validación de lo que escribe el usuario (rangos del enunciado)
# ---------------------------------------------------------------------------
def validar(caso, formulario):
    valores, errores = {}, []
    for campo in caso["campos"]:
        texto = formulario.get(campo["nombre"], "").strip().replace(",", ".")
        etiqueta = campo["etiqueta"].lower()
        try:
            numero = float(texto)
        except ValueError:
            errores.append(f"Escribe un número válido en «{etiqueta}».")
            continue
        if campo["tipo"] == "opcion":
            if numero not in (0, 1):
                errores.append(f"En «{etiqueta}» elige una de las opciones.")
                continue
        elif not campo["min"] <= numero <= campo["max"]:
            if campo["unidad"] == "S/":
                rango = f"S/ {campo['min']} y S/ {campo['max']}"
            else:
                unidad = f" {campo['unidad']}" if campo["unidad"] else ""
                rango = f"{campo['min']} y {campo['max']}{unidad}"
            errores.append(f"«{campo['etiqueta']}» debe estar entre {rango}.")
            continue
        valores[campo["nombre"]] = numero
    return valores, errores


# ---------------------------------------------------------------------------
# 4. Textos explicativos (se calculan con las métricas reales de modelo.json)
# ---------------------------------------------------------------------------
def textos(clave):
    m = MODELOS[clave]
    c = m["coeficientes"]
    mae, r2, media = m["mae"], m["r2"], m["media_objetivo"]
    pct_mae = mae / media * 100

    if clave == "caso-01":
        coef = [
            ("Intercepto", f"{m['intercepto']:.2f}",
             "Pedidos base cuando todas las variables valen 0."),
            ("Consultas web", f"{c['consultas_web']:+.3f}",
             f"Cada consulta adicional se asocia con {c['consultas_web']:.2f} pedidos más "
             f"(unas {1 / c['consultas_web']:.0f} consultas más ≈ 1 pedido más)."),
            ("Inversión en publicidad", f"{c['inversion_publicidad_soles']:+.4f}",
             f"Cada S/ 100 adicionales se asocian con {c['inversion_publicidad_soles'] * 100:.2f} "
             f"pedidos más: el efecto existe, pero es pequeño."),
            ("Campaña escolar", f"{c['campana_escolar']:+.2f}",
             f"Si hay campaña escolar, se esperan unos {c['campana_escolar']:.1f} pedidos más "
             f"que sin campaña, con las otras variables iguales."),
        ]
        metricas = [
            ("MAE", f"{mae:.2f} pedidos",
             f"En promedio, la estimación se equivoca por unos {mae:.1f} pedidos por semana. "
             f"Frente a un promedio de {media:.1f} pedidos, es un error de ≈ {pct_mae:.0f} %."),
            ("R²", f"{r2:.3f}",
             f"El modelo explica ≈ {r2 * 100:.0f} % de la variación de los pedidos; "
             f"el {100 - r2 * 100:.0f} % restante depende de factores que no están en el CSV."),
        ]
        explicacion = [
            "El caso busca anticipar cuántos kits de robótica pedirán los clientes cada semana.",
            f"Usé el CSV con {m['n_total']} semanas simuladas; no tiene ausentes ni duplicados.",
            "Las entradas X son consultas web, inversión en publicidad (S/) y campaña escolar (0 o 1).",
            "La respuesta y es el número de pedidos de kits de esa semana.",
            f"Separé 80 % para entrenar ({m['n_entrenamiento']} semanas) y 20 % para probar ({m['n_prueba']} semanas).",
            "Entrené una regresión lineal múltiple con Scikit-learn (LinearRegression).",
            "El modelo aprendió un intercepto y un coeficiente por variable, guardados en modelo.json.",
            f"En las semanas de prueba, el MAE fue {mae:.2f} pedidos y el R² fue {r2:.2f}.",
            "Flask lee modelo.json, valida los rangos y calcula la estimación al pulsar «Calcular».",
            "La página muestra una advertencia: es una estimación con datos simulados, no una garantía.",
        ]
        conclusiones = [
            "Las consultas web son la variable que más pesa: sirven como señal temprana de demanda.",
            "La campaña escolar suma pedidos de forma clara; conviene planificar el stock antes de cada campaña.",
            "La publicidad aporta poco por cada sol invertido; con estos datos no justifica por sí sola grandes aumentos de presupuesto.",
            f"Un R² de {r2:.2f} indica un buen ajuste pero no perfecto: queda variación que el modelo no explica.",
        ]
        limites = [
            "Los datos son simulados, no registros reales de una tienda.",
            "Solo hay 110 semanas y 22 de prueba: con otra separación, el MAE y el R² cambiarían un poco.",
            "El modelo supone relaciones en línea recta y no conoce feriados, stock agotado, precios ni competencia.",
            "Fuera de los rangos entrenados (5–70 consultas, S/ 0–1100) la estimación no es confiable; por eso la web los bloquea.",
            "No debe usarse para comprar inventario sin verificarla con la experiencia del negocio.",
        ]
        sentido = {
            0: "Es un valor bajo y coherente: pocas consultas, poca publicidad y sin campaña.",
            1: "Es un valor alto y coherente: muchas consultas, publicidad fuerte y campaña activa.",
        }
    else:
        coef = [
            ("Intercepto", f"S/ {m['intercepto']:.2f}",
             "Costo base cuando la distancia y la carga valen 0 y la vía es asfaltada."),
            ("Distancia (km)", f"{c['distancia_km']:+.3f}",
             f"Cada kilómetro adicional suma ≈ S/ {c['distancia_km']:.2f} al costo."),
            ("Carga (kg)", f"{c['carga_kg']:+.4f}",
             f"Cada kg adicional suma ≈ S/ {c['carga_kg']:.3f} (≈ S/ {c['carga_kg'] * 100:.2f} por cada 100 kg)."),
            ("Vía no asfaltada", f"{c['via_no_asfaltada']:+.2f}",
             f"Una vía no asfaltada suma ≈ S/ {c['via_no_asfaltada']:.2f} frente a una asfaltada, con distancia y carga iguales."),
        ]
        metricas = [
            ("MAE", f"S/ {mae:.2f}",
             f"En promedio, la estimación se equivoca por ≈ S/ {mae:.2f} por viaje. "
             f"Frente a un costo promedio de S/ {media:.0f}, es un error de ≈ {pct_mae:.1f} %."),
            ("R²", f"{r2:.3f}",
             f"El modelo explica ≈ {r2 * 100:.1f} % de la variación del costo; "
             f"casi todo el precio depende de estas tres variables."),
        ]
        explicacion = [
            "El caso busca estimar cuánto cuesta un viaje de transporte agrícola antes de realizarlo.",
            f"Usé el CSV con {m['n_total']} viajes simulados; no tiene ausentes ni duplicados.",
            "Las entradas X son distancia (km), carga (kg) y vía no asfaltada (0 = asfaltada, 1 = no asfaltada).",
            "La respuesta y es el costo del transporte en soles.",
            f"Separé 80 % para entrenar ({m['n_entrenamiento']} viajes) y 20 % para probar ({m['n_prueba']} viajes).",
            "Entrené una regresión lineal múltiple con Scikit-learn (LinearRegression).",
            "El modelo aprendió un intercepto y un coeficiente por variable, guardados en modelo.json.",
            f"En los viajes de prueba, el MAE fue S/ {mae:.2f} y el R² fue {r2:.2f}.",
            "Flask lee modelo.json, valida los rangos y calcula el costo al pulsar «Calcular».",
            "La página muestra una advertencia: es una estimación con datos simulados, no una cotización.",
        ]
        conclusiones = [
            "La distancia es el factor que más pesa en el costo; la carga y el tipo de vía ajustan el monto.",
            "Una vía no asfaltada encarece el viaje en un monto fijo, útil para cotizar rutas de trocha.",
            f"Un R² de {r2:.2f} indica un ajuste muy bueno: las tres variables explican casi todo el costo simulado.",
            f"Con un MAE de S/ {mae:.0f}, la estimación sirve como referencia rápida, no como precio final.",
        ]
        limites = [
            "Los datos son simulados, no tarifas reales de transportistas.",
            "Solo hay 130 viajes y 26 de prueba: con otra separación, las métricas variarían un poco.",
            "El modelo supone que la vía suma un monto fijo; en la realidad el efecto podría crecer con la distancia.",
            "No incluye combustible, peajes, clima, tipo de producto ni tiempos de espera.",
            "Fuera de los rangos entrenados (3–165 km, 50–2050 kg) la estimación no es confiable; por eso la web los bloquea.",
        ]
        sentido = {
            0: "Es un costo bajo y coherente: viaje corto, carga liviana y ruta asfaltada.",
            1: "Es un costo alto y coherente: viaje largo, carga pesada y vía no asfaltada.",
        }
    return coef, metricas, explicacion, conclusiones, limites, sentido


# ---------------------------------------------------------------------------
# 5. Rutas
# ---------------------------------------------------------------------------
@app.route("/")
def inicio():
    return redirect(url_for("caso", clave="caso-01"))


@app.route("/<clave>", methods=["GET", "POST"])
def caso(clave):
    if clave not in CASOS:
        abort(404)
    datos_caso = CASOS[clave]
    modelo = MODELOS[clave]

    escritos = {c["nombre"]: c["ejemplo"] for c in datos_caso["campos"]}  # valores de ejemplo
    resultado, errores = None, []

    # POST = el usuario pulsó el botón «Calcular».
    if request.method == "POST":
        escritos = {c["nombre"]: request.form.get(c["nombre"], "") for c in datos_caso["campos"]}
        valores, errores = validar(datos_caso, request.form)
        if not errores:
            estimado = predecir(clave, valores)
            resultado = {"texto": formatear(clave, estimado), "exacto": f"{estimado:.2f}"}

    # Pruebas obligatorias del enunciado, calculadas con el mismo modelo.
    coef, metricas, explicacion, conclusiones, limites, sentido = textos(clave)
    pruebas = []
    for i, p in enumerate(datos_caso["pruebas"]):
        est = predecir(clave, p["valores"])
        pruebas.append({"nombre": p["nombre"], "resultado": formatear(clave, est),
                        "exacto": f"{est:.2f}", "sentido": sentido[i]})

    otra_clave = "caso-02" if clave == "caso-01" else "caso-01"
    return render_template(
        "index.html", clave=clave, caso=datos_caso, modelo=modelo, escritos=escritos,
        resultado=resultado, errores=errores, pruebas=pruebas, coef=coef, metricas=metricas,
        explicacion=explicacion, conclusiones=conclusiones, limites=limites,
        otro=CASOS[otra_clave], otra_clave=otra_clave,
    )


@app.route("/<clave>/<archivo>")
def descargar(clave, archivo):
    # Permite abrir datos.csv y modelo.json de cada caso desde la página.
    if clave not in CASOS or archivo not in ("datos.csv", "modelo.json"):
        abort(404)
    return send_from_directory(RAIZ / CASOS[clave]["carpeta"], archivo)


if __name__ == "__main__":
    # debug solo para practicar en la computadora.
    app.run(debug=True)
