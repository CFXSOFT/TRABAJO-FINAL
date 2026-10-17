# casos.py | Configuración compartida de los dos casos del taller (SENATI).
# Lo usan tanto entrenar.py (para entrenar) como app.py (para la web),
# así los rangos y nombres de columnas son siempre los mismos.

CASOS = {
    "caso-01": {
        "numero": "01",
        "carpeta": "caso_01_pedidos_kits",
        "corto": "Pedidos de kits",
        "titulo": "Estimación de pedidos de kits de robótica",
        "pregunta": "¿Cuántos kits de robótica se pedirán esta semana?",
        "contexto": ("Una tienda educativa quiere anticipar sus pedidos semanales. "
                     "Más consultas web y más inversión en publicidad pueden asociarse con más ventas; "
                     "la campaña escolar se marca con 1 (hay campaña) o 0 (no hay)."),
        "objetivo": "pedidos_kits",
        "unidad_objetivo": "pedidos por semana",
        "campos": [
            {"nombre": "consultas_web", "etiqueta": "Consultas web de la semana",
             "unidad": "consultas", "tipo": "numero", "min": 5, "max": 70, "paso": 1,
             "ejemplo": 20},
            {"nombre": "inversion_publicidad_soles", "etiqueta": "Inversión en publicidad",
             "unidad": "S/", "tipo": "numero", "min": 0, "max": 1100, "paso": 1,
             "ejemplo": 200},
            {"nombre": "campana_escolar", "etiqueta": "¿Hay campaña escolar?",
             "unidad": "", "tipo": "opcion", "min": 0, "max": 1,
             "opciones": [(0, "No (0)"), (1, "Sí (1)")], "ejemplo": 0},
        ],
        "pruebas": [
            {"nombre": "Semana con 20 consultas, S/ 200 y sin campaña",
             "valores": {"consultas_web": 20, "inversion_publicidad_soles": 200, "campana_escolar": 0}},
            {"nombre": "Semana con 55 consultas, S/ 800 y con campaña",
             "valores": {"consultas_web": 55, "inversion_publicidad_soles": 800, "campana_escolar": 1}},
        ],
    },
    "caso-02": {
        "numero": "02",
        "carpeta": "caso_02_transporte_agricola",
        "corto": "Transporte agrícola",
        "titulo": "Estimación del costo de transporte agrícola",
        "pregunta": "¿Cuánto costará este viaje de transporte agrícola?",
        "contexto": ("Un transportista quiere estimar el costo de un viaje a partir de la distancia, "
                     "el peso de la carga y el tipo de vía. Los datos son simulados, inspirados en el "
                     "transporte agrícola regional."),
        "objetivo": "costo_transporte_soles",
        "unidad_objetivo": "soles por viaje",
        "campos": [
            {"nombre": "distancia_km", "etiqueta": "Distancia del viaje",
             "unidad": "km", "tipo": "numero", "min": 3, "max": 165, "paso": "any",
             "ejemplo": 25},
            {"nombre": "carga_kg", "etiqueta": "Peso de la carga",
             "unidad": "kg", "tipo": "numero", "min": 50, "max": 2050, "paso": "any",
             "ejemplo": 400},
            {"nombre": "via_no_asfaltada", "etiqueta": "Tipo de vía",
             "unidad": "", "tipo": "opcion", "min": 0, "max": 1,
             "opciones": [(0, "Asfaltada (0)"), (1, "No asfaltada (1)")], "ejemplo": 0},
        ],
        "pruebas": [
            {"nombre": "Viaje de 25 km, 400 kg, ruta asfaltada",
             "valores": {"distancia_km": 25, "carga_kg": 400, "via_no_asfaltada": 0}},
            {"nombre": "Viaje de 100 km, 1200 kg, ruta no asfaltada",
             "valores": {"distancia_km": 100, "carga_kg": 1200, "via_no_asfaltada": 1}},
        ],
    },
}
