# tennispredictor
Tennis predictor
import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.special import expit
from openai import OpenAI
import warnings
import wikipedia

warnings.filterwarnings('ignore')
wikipedia.set_lang("es")

class TennisChatbot:
    def __init__(self):
        self.conversation_active = True
        wikipedia.set_lang("es")

    def greet(self):
        print("🎾 ¡Hola! Soy un chatbot experto en tenis.")
        print("\nDime qué quieres hacer:")
        print("1. Consultar información sobre tenis en Wikipedia")
        print("2. Realizar un pronóstico de tenis")
        print("Escribe 'salir' para terminar")

    def consultar_wikipedia(self, consulta):
        try:
            titulos = wikipedia.search(consulta, results=3)
            if not titulos:
                return "❌ No encontré información relacionada"

            titulo = titulos[0]
            resumen = wikipedia.summary(titulo, sentences=5, auto_suggest=True, redirect=True)
            resumen_lower = resumen.lower()

            if "tenis" not in resumen_lower and "tenista" not in resumen_lower:
                return "❌ Esto no es tenis, pregunta sobre temas relacionados con el tenis"

            return resumen
        except wikipedia.DisambiguationError as e:
            return f"⚠️ Tu consulta es ambigua. Quizás te refieres a: {e.options[:5]}"
        except wikipedia.PageError:
            return "❌  No encontré información relacionada"
        except Exception as e:
            return f"⚠️ Ocurrió un error inesperado: {e}"

    def handle_wikipedia(self):
        print("\n📚 Dime qué tema exacto quieres consultar")
        while True:
            user_input = input("✍️ Escribe tu pregunta o 'volver': ").strip()
            if user_input.lower() == "volver":
                break
            if user_input.lower() == "salir":
                self.conversation_active = False
                break
            respuesta = self.consultar_wikipedia(user_input)
            print(f"\n📖 Resultado:\n{respuesta}\n")

    def run_prediction(self):
        print("\n📊 Vamos a realizar un pronóstico del partido")
        print("Introduce los datos del partido\n")
        try:
            calcular_ev_tenis()
        except NameError:
            print("❌ No puedo realizar el pronóstico")
        except Exception as e:
            print(f"⚠️ Hubo un error técnico: {e}")

    def start_chat(self):
        self.greet()
        while self.conversation_active:
            opcion = input("\nSelecciona una opción (1, 2) o escribe 'salir': ").strip().lower()

            if opcion == "salir":
                print("👋 ¡Gracias por usar el chatbot! Hasta pronto.")
                break
            elif opcion == "1":
                self.handle_wikipedia()
            elif opcion == "2":
                self.run_prediction()
            else:
                print("❗ Opción no válida. Elige 1, 2 o 'salir'.")


# ==================== FUNCIONES DE CÁLCULO ====================
def calcular_puntuacion_ponderada(jugador, cuota_jugador, superficie, es_challenger):
    """Calcula la puntuación ponderada según la fórmula propuesta"""
    peso_elo = 0.45
    peso_especializacion = 0.25
    peso_momentum = 0.25
    peso_local = 0.05
    
    # Bonificación para Challengers con momentum:
    if es_challenger and jugador.get('gano_sin_ceder_sets', False):
        peso_momentum += 0.1  # Bonus adicional

    elo_norm = jugador['elo'] / 2500
    
    componente_elo = peso_elo * elo_norm
    componente_especializacion = peso_especializacion * (1 if jugador['es_especialista'] else 0)
    componente_momentum = peso_momentum * (1 if jugador.get('gano_sin_ceder_sets', False) else 0)
    componente_local = peso_local * (1 if jugador.get('es_local', False) else 0)
    
    puntuacion = (componente_elo + componente_especializacion + componente_momentum + componente_local)
    
    if es_challenger and cuota_jugador > 1.83 and puntuacion > 0.6:
        puntuacion *= 1.25
            
    return puntuacion

def calcular_ratio_victorias(jugador):
    """Calcula el ratio de victorias sobre partidos totales"""
    total = jugador['victorias_sup'] + jugador['derrotas_sup']
    return jugador['victorias_sup'] / total if total > 0 else 0.5

def calcular_peso_ratio_challenger(jugador1, jugador2, superficie):
    """
    Calcula el peso del ratio victorias/partidos según la fórmula especificada
    para Challengers con baja puntuación predictiva (<0.6)
    """
    peso = 0.4
    
    # Calcular ratios y partidos totales
    partidos_j1 = jugador1['victorias_sup'] + jugador1['derrotas_sup']
    partidos_j2 = jugador2['victorias_sup'] + jugador2['derrotas_sup']
    ratio_j1 = calcular_ratio_victorias(jugador1)
    ratio_j2 = calcular_ratio_victorias(jugador2)
    
    # Añadir bonificaciones para jugador
    if (partidos_j1 >= 15) or (partidos_j2 >= 15):
        peso += 0.2
    if (jugador1['es_especialista'] or jugador2['es_especialista']) and superficie == 'dura':
        peso += 0.2
    if (ratio_j1 >= 0.65) or (ratio_j2 >=0.65):
        peso += 0.1
        
    # Aplicar penalizaciones para jugador
    if (partidos_j1 < 8) or (partidos_j2 < 8):
        peso -= 0.25
    
    # Asegurar que el peso esté dentro de límites razonables
    return max(0.1, min(0.7, peso))

def manejar_no_aplica(jugador1, jugador2, cuota1, cuota2, ev1, ev2):
    """
    Maneja el caso cuando se selecciona 'No Aplica' para algun jugador,
    asignando probabilidades basadas en las cuotas.
    """
    no_aplica_jugador1 = jugador1.get('no_aplica_sets', False)
    no_aplica_jugador2 = jugador2.get('no_aplica_sets', False)

    # Determinar favorito por cuotas
    if no_aplica_jugador1 and no_aplica_jugador2:
        if cuota1 < cuota2:   
            ev1, ev2 = (ev1 * 1.55), (ev2 * 0.35)
        elif cuota2 < cuota1:  
            ev1, ev2 = (ev1 * 0.35), (ev2 * 1.55)

    elif no_aplica_jugador1:
        if cuota1 < cuota2:   
            ev1, ev2 = (ev1 * 0.35), (ev2 * 1.55)
    
    elif no_aplica_jugador2:
        if cuota2 < cuota1:   
            ev1, ev2 = (ev1 * 1.55), (ev2 * 0.35)           
            
    return ev1, ev2
    
def calcular_prob_cuotas(cuota1, cuota2, elo1, elo2, ranking1, ranking2, jugador1, jugador2, superficie, es_challenger):
    """Calcula probabilidades ajustadas considerando la fórmula ponderada"""
    puntuacion1 = calcular_puntuacion_ponderada(jugador1, cuota1, superficie, es_challenger)
    puntuacion2 = calcular_puntuacion_ponderada(jugador2, cuota2, superficie, es_challenger)
    
    prob_impl_cuota1 = 1 / cuota1
    prob_impl_cuota2 = 1 / cuota2
    
    factor_ajuste1 = min(puntuacion1 / (puntuacion1 + puntuacion2) * 2, 1.5) * (1.1 if jugador1.get('gano_sin_ceder_sets', False) else 1.0)  
    factor_ajuste2 = min(puntuacion2 / (puntuacion1 + puntuacion2) * 2, 1.5) * (1.1 if jugador2.get('gano_sin_ceder_sets', False) else 1.0)
    
    prob_ajustada1 = prob_impl_cuota1 * factor_ajuste1
    prob_ajustada2 = prob_impl_cuota2 * factor_ajuste2
    
    suma_prob = prob_ajustada1 + prob_ajustada2
    
    # Aplicar ajuste específico para Challengers con baja puntuación
    if es_challenger and (puntuacion1 < 0.6 or puntuacion2 < 0.6):
        peso_ratio = calcular_peso_ratio_challenger(jugador1, jugador2, superficie)
        prob_ajustada1 = (prob_ajustada1 * (1 - peso_ratio)) + (calcular_ratio_victorias(jugador1) * peso_ratio)
        prob_ajustada2 = (prob_ajustada2 * (1 - peso_ratio)) + (calcular_ratio_victorias(jugador2) * peso_ratio)
        suma_prob = prob_ajustada1 + prob_ajustada2
    
    return prob_ajustada1 / suma_prob, prob_ajustada2 / suma_prob

def predecir_flagwin(jugador1, jugador2, indice_j1, indice_j2, diff_indice, cuota1, cuota2, 
                     diff_cuotas, diff_elo, es_especialista_1, es_especialista_2,
                     momentum_1, momentum_2, h2h1, h2h2, no_aplica_jugador1, no_aplica_jugador2):

    diff_elo = abs(jugador1['elo'] - jugador2['elo'])

    # Determinar jugador favorito
    if indice_j1 > indice_j2:
        jugador_fav = jugador1
        indice_fav = indice_j1
        es_especialista_fav = es_especialista_1
        momentum_fav = momentum_1
        h2h_fav = h2h1
        cuota_fav = cuota1
    else:
        jugador_fav = jugador2
        indice_fav = indice_j2
        es_especialista_fav = es_especialista_2
        momentum_fav = momentum_2
        h2h_fav = h2h2
        cuota_fav = cuota2

    # Árbol de decisión
    if indice_fav >= 0.6:
        if diff_indice >= 0.3:
            flagwin = 1
        else:
            flagwin = 2 if diff_cuotas >= 1.16 else (1 if es_especialista_fav else 0)
    elif 0.4 <= indice_fav < 0.6:
        if diff_elo > 100:
            flagwin = 2 if diff_cuotas > 1.03 else (1 if momentum_fav else 0)
        else:
            flagwin = 1 if h2h_fav > 0 else 0
    else:
        flagwin = 1 if (diff_elo > 100 and momentum_fav) or (h2h_fav > 0) else 0

    # Si ambos son "no aplica", decidir por cuotas
    if no_aplica_jugador1 and no_aplica_jugador2:
        flagwin = 1 if cuota1 < cuota2 else 2
    elif no_aplica_jugador1 and cuota1 < cuota2:
        flagwin = 2
    elif no_aplica_jugador2 and cuota2 < cuota1:
        flagwin = 1

    if flagwin not in [0, 1, 2]:
        flagwin = 1 if cuota1 < cuota2 else 2  # Por defecto, favorito por cuotas
    
    return flagwin, indice_fav

    return flagwin, indice_fav

# ==================== FUNCIONES DE ANÁLISIS ====================

def mostrar_analisis_formula(jugador1, jugador2, puntuacion1, puntuacion2):
    """Muestra el análisis detallado de la fórmula ponderada"""
    print("\n🔍 ANÁLISIS FÓRMULA PONDERADA:")
    print(f"● {jugador1['nombre']}: Puntuación = {puntuacion1:.2f}/1.0")
    print(f"● {jugador2['nombre']}: Puntuación = {puntuacion2:.2f}/1.0")
    
    if puntuacion1 > puntuacion2 + 0.15:
        print(f"\n✅ La fórmula favorece claramente a {jugador1['nombre']}")
    elif puntuacion2 > puntuacion1 + 0.15:
        print(f"\n✅ La fórmula favorece claramente a {jugador2['nombre']}")
    else:
        print("\n⚠️ La fórmula no muestra un favorito claro, considerar otros factores")

def calcular_ratio_rendimiento(victorias, derrotas):
    """Calcula el ratio de rendimiento (victorias/total)"""
    total = victorias + derrotas
    return victorias / total if total > 0 else 0.5

def calcular_factor_superficie(victorias, derrotas):
    """Calcula el factor de superficie basado en el rendimiento"""
    ratio = calcular_ratio_rendimiento(victorias, derrotas)
    if ratio <= 0.4: return 0.45
    if 0.4 < ratio <= 0.5: return 0.55
    return 1.0

def prob_ranking_elo(ranking1, ranking2, elo1, elo2):
    """Calcula probabilidad combinada basada en ranking y ELO"""
    prob_ranking = 1 / (1 + (ranking1 / ranking2))
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 125))
    return (0.65 * prob_elo) + (0.35 * prob_ranking)

def calcular_impacto_especialista(jugador1, jugador2):
    """Calcula el impacto de ser especialista en superficie"""
    if jugador1['es_especialista'] and not jugador2['es_especialista']:
        return 0.25  
    elif jugador2['es_especialista'] and not jugador1['es_especialista']:
        return -0.25  
    return 0  

def calcular_indice_emocional_challenger(jugador, cuota):
    """Calcula índice emocional específico para torneos Challenger"""
    victorias = jugador['victorias_sup']
    derrotas = jugador['derrotas_sup']
    total = victorias + derrotas
    ratio_vict = victorias / total if total > 0 else 0.5
    
    gano_sin_ceder = jugador.get('gano_sin_ceder_sets', False)
    sets_decisivos = 1 if not gano_sin_ceder else 0
    underdog = 1 if cuota > 1.83 else 0
    local = 1 if jugador.get('es_local', False) else 0
    especialista = 1 if jugador.get('es_especialista', False) else 0
    
    indice_emocional = (0.35 * ratio_vict + 0.15 * sets_decisivos + 
                        0.1 * underdog + 0.25 * local + 0.15 * especialista)

    if (victorias + derrotas) <= 2:
        indice_emocional *= 0.75
            
    return indice_emocional

def calcular_bonificacion_estrategica(jugador, oponente, superficie, cuota, h2h_ventaja, es_local, ev):
    """Calcula bonificación estratégica basada en combinaciones clave de variables"""
    bonificacion = 0
   
    # Combinación 1: ELO/Ranking superior + Especialista + Momentum
    cumple_elo_ranking = (jugador['elo'] > oponente['elo'] + 50 or jugador['ranking'] < oponente['ranking'] + 10)
    es_especialista = jugador['es_especialista']
    sin_ceder_sets = jugador.get('gano_sin_ceder_sets', False)
    
    if cumple_elo_ranking and es_especialista:
        if sin_ceder_sets:
            bonificacion += 0.3
            print(f"⚠️ Bonificación +0.3 para {jugador['nombre']} (ELO/Ranking + Especialista + Sin ceder sets)")
        else:
            bonificacion += 0.2
            print(f"⚠️ Bonificación +0.15 para {jugador['nombre']} (ELO/Ranking + Especialista)")

    # Combinación 2: Ventaja H2H + Localía
    if h2h_ventaja and es_local:
        bonificacion += 0.15
        print(f"⚠️ Bonificación +0.2 para {jugador['nombre']} (Ventaja H2H + Localía)")
    
    # Combinación 3: Sorpresa reciente + Ratio positivo + Cuota baja
    if jugador['ajuste_sorpresa'] == 0.91 and \
       calcular_ratio_victorias(jugador) > 0.6 and cuota < 1.95:
        bonificacion += 0.15
        print(f"⚠️ Bonificación +0.15 para {jugador['nombre']} (Sorpresa + Ratio + Cuota baja)")

    return bonificacion    

# ==================== FUNCIONES DE INTERFAZ ====================

def mostrar_titulo(titulo):
    """Muestra un título formateado"""
    print("\n" + "="*50)
    print(f"=== {titulo.upper()} ===")
    print("="*50)

def obtener_input_validado(mensaje, tipo, rango=None, opciones=None, paso=None, modo_edicion=False, valor_actual=None):
    """
    Obtiene entrada del usuario con validación.
    
    Args:
        mensaje (str): Mensaje a mostrar al usuario
        tipo (type): Tipo de dato esperado (int, float, str)
        rango (tuple): (min, max) para valores numéricos
        opciones (list): Opciones válidas para strings
        paso (str): Identificador del paso actual para edición
        modo_edicion (bool): Indica si estamos en modo edición
        valor_actual: Valor actual del campo a editar
    
    Returns:
        Valor validado ingresado por el usuario
    """
    while True:
        try:
            if modo_edicion:
                entrada = input(f"[EDITANDO] {mensaje} (o 'x' para mantener actual): ").strip()
                if entrada.lower() == 'x':
                    return valor_actual
            else:
                entrada = input(mensaje).strip()
            
            if tipo == float:
                valor = float(entrada.replace(',', '.'))
            elif tipo == int:
                valor = int(entrada)
            else:
                valor = entrada
            
            if opciones:
                entrada_normalizada = valor.lower().replace('í', 'i').replace('á', 'a')
                opciones_normalizadas = [opcion.lower().replace('í', 'i').replace('á', 'a') for opcion in opciones]
                
                if entrada_normalizada not in opciones_normalizadas:
                    raise ValueError(f"Opción no válida. Debe ser una de: {opciones}")
                return opciones[opciones_normalizadas.index(entrada_normalizada)]
            
            if rango and (valor < rango[0] or valor > rango[1]):
                raise ValueError(f"El valor debe estar entre {rango[0]} y {rango[1]}")
            
            return valor
        
        except ValueError as e:
            print(f"⚠️ Error: {e}. Por favor, introduce un dato correcto.\n")

def ingresar_datos_jugador(num_jugador, superficie, torneo, modo_edicion=False, jugador_actual=None):
    """Recoge los datos de un jugador mediante input del usuario"""
    print(f"\n=== DATOS DEL JUGADOR {num_jugador} ===")
    
    if modo_edicion and jugador_actual:
        datos = jugador_actual.copy()
    else:
        datos = {
            'nombre': '',
            'elo': 1500,
            'ranking': 100,
            'victorias_sup': 0,
            'derrotas_sup': 0,
            'es_especialista': False,
            'ajuste_sorpresa': 1.0,
            'gano_sin_ceder_sets': False,
            'es_local': False
        }

    datos['nombre'] = obtener_input_validado(
        f"Nombre del Jugador {num_jugador}: ",
        tipo=str,
        modo_edicion=modo_edicion,
        valor_actual=datos.get('nombre', '')
    )

    datos['elo'] = obtener_input_validado(
        "ELO: ",
        tipo=float,
        rango=(1000, 2500),
        modo_edicion=modo_edicion,
        valor_actual=datos.get('elo', 1500)
    )

    datos['ranking'] = obtener_input_validado(
        "Ranking ATP: ",
        tipo=int,
        rango=(1, 5000),
        modo_edicion=modo_edicion,
        valor_actual=datos.get('ranking', 100)
    )

    datos['victorias_sup'] = obtener_input_validado(
        f"Victorias en {superficie} (este año): ",
        tipo=int,
        rango=(0, 100),
        modo_edicion=modo_edicion,
        valor_actual=datos.get('victorias_sup', 0)
    )

    datos['derrotas_sup'] = obtener_input_validado(
        f"Derrotas en {superficie} (este año): ",
        tipo=int,
        rango=(0, 100),
        modo_edicion=modo_edicion,
        valor_actual=datos.get('derrotas_sup', 0)
    )

    es_especialista = obtener_input_validado(
        f"¿Es especialista en {superficie}? (Sí/No): ",
        tipo=str,
        opciones=['Sí', 'No', 'si', 'no', 's', 'n'],
        modo_edicion=modo_edicion,
        valor_actual='si' if datos.get('es_especialista', False) else 'no'
    ).lower() in ['sí', 'si', 's']
    datos['es_especialista'] = es_especialista

    if not modo_edicion or input("[EDITANDO] ¿Cambiar ajuste sorpresa? (s/n): ").lower() == 's':
        respuesta_sorpresa = obtener_input_validado(
            f"¿Viene de derrotar en el {torneo} a un jugador con +10 puestos mejor en ranking? (Sí/No): ",
            tipo=str,
            opciones=['Sí', 'No', 'si', 'no', 's', 'n'],
            modo_edicion=modo_edicion,
            valor_actual='si' if datos.get('ajuste_sorpresa', 1.0) == 0.91 else 'no'
        ).lower() in ['sí', 'si', 's']
        datos['ajuste_sorpresa'] = 0.91 if respuesta_sorpresa == 'si' else 1.0

    respuesta = obtener_input_validado(
        f"¿Ganó sin ceder sets su último partido en el {torneo}? (Sí/No/No Aplica): ",
        tipo=str,
        opciones=['Sí', 'No', 'si', 'no', 's', 'n', 'no aplica', 'No Aplica'],
        modo_edicion=modo_edicion,
        valor_actual='si' if datos.get('gano_sin_ceder_sets', False)
        else 'no aplica' if datos.get('no_aplica_sets', False)
                else 'no'
    )

    if not isinstance(respuesta, str):
        raise ValueError("La respuesta debe ser un string")

# Procesamiento seguro
    respuesta_procesada = str(respuesta).lower().strip()

    datos['no_aplica_sets'] = respuesta_procesada in ['no aplica', 'n/a']
    if datos['no_aplica_sets']:
        datos['gano_sin_ceder_sets'] = None
    else:
        datos['gano_sin_ceder_sets'] = respuesta_procesada in ['sí', 'si', 's']

    es_local = obtener_input_validado(
        "¿Juega como local? (Sí/No): ",
        tipo=str,
        opciones=['Sí', 'No', 'si', 'no', 's', 'n'],
        modo_edicion=modo_edicion,
        valor_actual='sí' if datos.get('es_local', False) else 'no'
    ).lower() in ['sí', 'si', 's']
    datos['es_local'] = es_local

    return datos

def editar_jugador(datos_ingresados, num_jugador):
    """Permite editar los datos de un jugador específico"""
    jugador_key = f'jugador{num_jugador}'
    jugador_actual = datos_ingresados.get(jugador_key, {})
    
    print(f"\nEditando datos del Jugador {num_jugador} ({jugador_actual.get('nombre', 'Nuevo Jugador')})")
    
    # Mostrar menú de edición
    print("\n1. Cambiar nombre")
    print("2. Cambiar ELO")
    print("3. Cambiar Ranking ATP")
    print("4. Cambiar victorias en superficie")
    print("5. Cambiar derrotas en superficie")
    print("6. Cambiar especialista")
    print("7. Cambiar ajuste sorpresa")
    print("8. Cambiar último partido sin ceder sets")
    print("9. Cambiar condición de local")
    print("10. Volver al menú principal")
    
    opcion = obtener_input_validado(
        "Seleccione una opción (1-10): ",
        tipo=int,
        rango=(1, 10)
    )
    
    if opcion == 1:
        datos_ingresados[jugador_key]['nombre'] = obtener_input_validado(
            f"Nuevo nombre del Jugador {num_jugador}: ",
            tipo=str,
            modo_edicion=True,
            valor_actual=jugador_actual.get('nombre', '')
        )
    elif opcion == 2:
        datos_ingresados[jugador_key]['elo'] = obtener_input_validado(
            "Nuevo ELO: ",
            tipo=float,
            rango=(1000, 2500),
            modo_edicion=True,
            valor_actual=jugador_actual.get('elo', 1500)
        )
    elif opcion == 3:
        datos_ingresados[jugador_key]['ranking'] = obtener_input_validado(
            "Nuevo Ranking ATP: ",
            tipo=int,
            rango=(1, 5000),
            modo_edicion=True,
            valor_actual=jugador_actual.get('ranking', 100)
        )
    elif opcion == 4:
        datos_ingresados[jugador_key]['victorias_sup'] = obtener_input_validado(
            f"Nuevas victorias en {datos_ingresados.get('superficie', 'superficie')} (este año): ",
            tipo=int,
            rango=(0, 100),
            modo_edicion=True,
            valor_actual=jugador_actual.get('victorias_sup', 0)
        )
    elif opcion == 5:
        datos_ingresados[jugador_key]['derrotas_sup'] = obtener_input_validado(
            f"Nuevas derrotas en {datos_ingresados.get('superficie', 'superficie')} (este año): ",
            tipo=int,
            rango=(0, 100),
            modo_edicion=True,
            valor_actual=jugador_actual.get('derrotas_sup', 0)
        )
    elif opcion == 6:
        es_especialista = obtener_input_validado(
            f"¿Es {jugador_actual.get('nombre', 'el jugador')} especialista en {datos_ingresados.get('superficie', 'esta superficie')}? (Sí/No): ",
            tipo=str,
            opciones=['Sí', 'No', 'si', 'no', 's', 'n'],
            modo_edicion=True,
            valor_actual='si' if jugador_actual.get('es_especialista', False) else 'no'
        ).lower() in ['sí', 'si', 's']
        datos_ingresados[jugador_key]['es_especialista'] = es_especialista
    elif opcion == 7:
        respuesta_sorpresa = obtener_input_validado(
            f"¿{jugador_actual.get('nombre', 'el jugador')} viene de derrotar en el {datos_ingresados.get('torneo', 'este torneo')} a un jugador con +10 puestos mejor en ranking? (Sí/No): ",
            tipo=str,
            opciones=['Sí', 'No', 'si', 'no', 's', 'n'],
            modo_edicion=True,
            valor_actual='si' if jugador_actual.get('ajuste_sorpresa', 1.0) == 0.91 else 'no'
        ).lower() in ['sí', 'si', 's']
        datos_ingresados[jugador_key]['ajuste_sorpresa'] = 0.91 if respuesta_sorpresa == 'si' else 1.0
    elif opcion == 8:
        gano_sin_ceder = obtener_input_validado(
            f"¿Ganó sin ceder sets su último partido en el {datos_ingresados.get('torneo', 'este torneo')}? (Sí/No/No Aplica): ",
            tipo=str,
            opciones=['Sí', 'No', 'si', 'no', 's', 'n', 'no aplica', 'No Aplica'],
            modo_edicion=True,
            valor_actual='sí' if jugador_actual.get('gano_sin_ceder_sets', False) else 'no'
        ).lower() in ['sí', 'si', 's']
        datos_ingresados[jugador_key]['gano_sin_ceder_sets'] = gano_sin_ceder
    elif opcion == 9:
        es_local = obtener_input_validado(
            f"¿Juega {jugador_actual.get('nombre', 'el jugador')} como local? (Sí/No): ",
            tipo=str,
            opciones=['Sí', 'No', 'si', 'no', 's', 'n'],
            modo_edicion=True,
            valor_actual='sí' if jugador_actual.get('es_local', False) else 'no'
        ).lower() in ['sí', 'si', 's']
        datos_ingresados[jugador_key]['es_local'] = es_local
    elif opcion == 10:
        return False  # Indica que no se deben guardar los cambios
    
    return True  # Indica que se deben guardar los cambios

def menu_edicion(datos_ingresados):
    """Muestra el menú de edición y maneja las opciones"""
    while True:
        print("\n=== MENÚ DE EDICIÓN ===")
        print("1. Cambiar superficie")
        print("2. Cambiar torneo")
        print("3. Cambiar tipo de torneo")
        print(f"4. Editar datos de ({datos_ingresados['jugador1'].get('nombre', 'Jugador 1')})")
        print(f"5. Editar datos de ({datos_ingresados['jugador2'].get('nombre', 'Jugador 2')})")
        print("6. Editar historial H2H")
        print("7. Editar cuotas")
        print("8. Continuar")
        print("9. Salir sin calcular")
        
        opcion = obtener_input_validado(
            "Seleccione una opción (1-9): ",
            tipo=int,
            rango=(1, 9)
        )
        
        if opcion == 1:
            datos_ingresados['superficie'] = obtener_input_validado(
                "Superficie (arcilla/hierba/dura/indoor): ",
                tipo=str,
                opciones=['arcilla', 'hierba', 'dura', 'indoor'],
                modo_edicion=True,
                valor_actual=datos_ingresados.get('superficie', 'arcilla')
            )
        elif opcion == 2:
            datos_ingresados['torneo'] = obtener_input_validado(
                "Escribe el nombre del torneo: ",
                tipo=str,
                modo_edicion=True,
                valor_actual=datos_ingresados.get('torneo', '')
            )
        elif opcion == 3:
            datos_ingresados['tipo_torneo'] = obtener_input_validado(
                "¿Qué tipo de torneo juega (ATP/challenger)? ",
                tipo=str,
                opciones=['ATP', 'challenger'],
                modo_edicion=True,
                valor_actual=datos_ingresados.get('tipo_torneo', 'ATP')
            )
        elif opcion == 4:
            if editar_jugador(datos_ingresados, 1):
                print("✅ Cambios guardados para Jugador 1")
        elif opcion == 5:
            if editar_jugador(datos_ingresados, 2):
                print("✅ Cambios guardados para Jugador 2")
        elif opcion == 6:
            if 'jugador1' in datos_ingresados and 'jugador2' in datos_ingresados:
                datos_ingresados['h2h1'] = obtener_input_validado(
                    f"Victorias de {datos_ingresados['jugador1'].get('nombre', 'Jugador 1')} vs {datos_ingresados['jugador2'].get('nombre', 'Jugador 2')}: ",
                    tipo=int,
                    rango=(0, 100),
                    modo_edicion=True,
                    valor_actual=datos_ingresados.get('h2h1', 0)
                )
                datos_ingresados['h2h2'] = obtener_input_validado(
                    f"Victorias de {datos_ingresados['jugador2'].get('nombre', 'Jugador 2')} vs {datos_ingresados['jugador1'].get('nombre', 'Jugador 1')}: ",
                    tipo=int,
                    rango=(0, 100),
                    modo_edicion=True,
                    valor_actual=datos_ingresados.get('h2h2', 0)
                )
        elif opcion == 7:
            if 'jugador1' in datos_ingresados and 'jugador2' in datos_ingresados:
                datos_ingresados['cuota1'] = obtener_input_validado(
                    f"Cuota para {datos_ingresados['jugador1'].get('nombre', 'Jugador 1')}: ",
                    tipo=float,
                    rango=(1.01, 50),
                    modo_edicion=True,
                    valor_actual=datos_ingresados.get('cuota1', 1.5)
                )
                datos_ingresados['cuota2'] = obtener_input_validado(
                    f"Cuota para {datos_ingresados['jugador2'].get('nombre', 'Jugador 2')}: ",
                    tipo=float,
                    rango=(1.01, 50),
                    modo_edicion=True,
                    valor_actual=datos_ingresados.get('cuota2', 1.5)
                )
        elif opcion == 8:
            return True  # Continuar con el cálculo
        elif opcion == 9:
            return False  # Salir sin calcular

# ==================== FUNCIÓN PRINCIPAL ====================
def calcular_ev_tenis():
    """Función principal que ejecuta la calculadora de EV para tenis"""
    datos_ingresados = {
        'superficie': None,
        'torneo': None,
        'tipo_torneo': None,
        'jugador1': {},
        'jugador2': {},
        'h2h1': None,
        'h2h2': None,
        'cuota1': None,
        'cuota2': None
    }
    
    while True:
        mostrar_titulo("calculadora de valor esperado (ev) para tenis")
        
        # Ingreso de datos básicos
        print("\n=== SUPERFICIE ===")
        superficie = obtener_input_validado(
            "Superficie (arcilla/hierba/dura/indoor): ",
            tipo=str,
            opciones=['arcilla', 'hierba', 'dura', 'indoor']
        )
        
        print("=== TORNEO ===")
        torneo = obtener_input_validado("Escribe el nombre del torneo: ", 
                                      tipo=str)

        print("\n=== TIPO DE TORNEO ===")
        tipo_torneo = obtener_input_validado(
            "¿Qué tipo de torneo juega (ATP/challenger)? ",
            tipo=str,
            opciones=['ATP', 'challenger']
        )
        es_challenger = tipo_torneo == "challenger"
        es_atp = tipo_torneo == "ATP"

        # Ingreso de datos de jugadores
        print("\n=== INGRESO DE DATOS ===")
        jugador1 = ingresar_datos_jugador(1, superficie, torneo)
        jugador2 = ingresar_datos_jugador(2, superficie, torneo)
        
        datos_ingresados.update({
            'jugador1': jugador1,
            'jugador2': jugador2,
            'superficie': superficie,
            'torneo': torneo,
            'tipo_torneo': tipo_torneo
        })

        # Historial H2H
        print("\n=== HISTORIAL (H2H) ===")
        h2h1 = obtener_input_validado(
            f"Victorias de {jugador1['nombre']} vs {jugador2['nombre']}: ",
            tipo=int,
            rango=(0, 100)
        )
        h2h2 = obtener_input_validado(
            f"Victorias de {jugador2['nombre']} vs {jugador1['nombre']}: ",
            tipo=int,
            rango=(0, 100)
        )
        total_h2h = h2h1 + h2h2
        
        if total_h2h >= 3:
            if h2h1 == total_h2h:
                prob_h2h1 = 0.945    
            elif h2h2 == total_h2h:
                prob_h2h1 = 0.055  
            else:
                prob_h2h1 = h2h1 / total_h2h
        else:
            prob_h2h1 = 0.5

        peso_h2h = 0.15 if total_h2h >= 1 else 0.0

        # Cuotas
        print("\n=== BOOKIES (CUOTAS) ===")
        cuota1 = obtener_input_validado(
            f"Cuota para {jugador1['nombre']}: ",
            tipo=float,
            rango=(1.01, 50)
        )
        cuota2 = obtener_input_validado(
            f"Cuota para {jugador2['nombre']}: ",
            tipo=float,
            rango=(1.01, 50)
        )
        
        datos_ingresados.update({
            'h2h1': h2h1,
            'h2h2': h2h2,
            'cuota1': cuota1,
            'cuota2': cuota2
        })

        # Confirmación antes de calcular
        print("\n¿Desea editar algún dato antes de calcular?")
        editar = obtener_input_validado(
            "Ingrese 'Sí' para editar o cualquier tecla para continuar: ",
            tipo=str
        ).lower()
        
        if editar in ('sí', 'si', 's'):
            continuar_calculo = menu_edicion(datos_ingresados)
            if not continuar_calculo:
                return

        # Actualizar variables con posibles cambios
        superficie = datos_ingresados['superficie']
        torneo = datos_ingresados['torneo']
        tipo_torneo = datos_ingresados['tipo_torneo']
        es_challenger = tipo_torneo == "challenger"
        es_atp = tipo_torneo == "ATP"
        jugador1 = datos_ingresados['jugador1']
        jugador2 = datos_ingresados['jugador2']
        h2h1 = datos_ingresados['h2h1']
        h2h2 = datos_ingresados['h2h2']
        cuota1 = datos_ingresados['cuota1']
        cuota2 = datos_ingresados['cuota2']
        total_h2h = h2h1 + h2h2

        # Cálculos intermedios
        factor_sup1 = calcular_factor_superficie(jugador1['victorias_sup'], jugador1['derrotas_sup'])
        factor_sup2 = calcular_factor_superficie(jugador2['victorias_sup'], jugador2['derrotas_sup'])

        prob_base_sin_ajuste = prob_ranking_elo(jugador1['ranking'], jugador2['ranking'], 
                                              jugador1['elo'], jugador2['elo'])
                                    
        prob_base1 = prob_base_sin_ajuste * factor_sup1 * jugador1['ajuste_sorpresa']
        prob_base2 = (1 - prob_base_sin_ajuste) * factor_sup2 * jugador2['ajuste_sorpresa']

        # Ajustes por localía
        if jugador1['es_local']:
            prob_base1 = min(prob_base1 + 0.075, 0.99)
            prob_base2 = 1 - prob_base1
        elif jugador2['es_local']:
            prob_base2 = min(prob_base2 + 0.075, 0.99)
            prob_base1 = 1 - prob_base2

        # Ajustes por rendimiento reciente
        victorias_total1 = jugador1['victorias_sup'] + jugador1['derrotas_sup']
        victorias_total2 = jugador2['victorias_sup'] + jugador2['derrotas_sup']
        
        if victorias_total1 >= 8 and jugador1['victorias_sup'] > (jugador1['derrotas_sup'] + 2):
            prob_base1 = min(prob_base1 * 1.25, 0.99)
        if victorias_total2 >= 8 and jugador2['victorias_sup'] > (jugador2['derrotas_sup'] + 2):
            prob_base2 = min(prob_base2 * 1.25, 0.99)

        # Cálculo de probabilidades desde cuotas
        prob_cuota1, prob_cuota2 = calcular_prob_cuotas(cuota1, cuota2, 
                                                      jugador1['elo'], jugador2['elo'], 
                                                      jugador1['ranking'], jugador2['ranking'],
                                                      jugador1, jugador2,
                                                      superficie, es_challenger)

        # Cálculo de forma actual
        if es_challenger:
            indice_emocional1 = calcular_indice_emocional_challenger(jugador1, cuota1)
            indice_emocional2 = calcular_indice_emocional_challenger(jugador2, cuota2)
            forma1 = 0.5 * (calcular_ratio_rendimiento(jugador1['victorias_sup'], jugador1['derrotas_sup'])) + 0.5 * indice_emocional1 
            forma2 = 0.5 * (calcular_ratio_rendimiento(jugador2['victorias_sup'], jugador2['derrotas_sup'])) + 0.5 * indice_emocional2
            prob_forma1 = forma1 / (forma1 + forma2) if (forma1 + forma2) > 0 else 0.5
            prob_forma2 = 1 - prob_forma1
        else:
            forma1 = calcular_ratio_rendimiento(jugador1['victorias_sup'], jugador1['derrotas_sup'])
            forma2 = calcular_ratio_rendimiento(jugador2['victorias_sup'], jugador2['derrotas_sup'])
            prob_forma1 = forma1 / (forma1 + forma2) if (forma1 + forma2) > 0 else 0.5
            prob_forma2 = 1 - prob_forma1

        # Ajustes por rachas positivas
        for jugador, prob_forma in [(jugador1, prob_forma1), (jugador2, prob_forma2)]:
            victorias_total = jugador['victorias_sup'] + jugador['derrotas_sup']
            if victorias_total >= 10 and jugador['victorias_sup'] >= (jugador['derrotas_sup'] + 2):
                if jugador == jugador1:
                    prob_forma1 = min(prob_forma * 1.45, 0.99)
                    prob_forma2 = 1 - prob_forma1
                else:
                    prob_forma2 = min(prob_forma * 1.45, 0.99)
                    prob_forma1 = 1 - prob_forma2
            elif victorias_total >= 7 and jugador['victorias_sup'] >= (jugador['derrotas_sup'] + 1):
                if jugador == jugador1:
                    prob_forma1 = min(prob_forma * 1.37, 0.99)
                    prob_forma2 = 1 - prob_forma1
                else:
                    prob_forma2 = min(prob_forma * 1.37, 0.99)
                    prob_forma1 = 1 - prob_forma2

        # Determinar pesos según diferencias
        diff_elo = abs(jugador1['elo'] - jugador2['elo'])
        diff_ranking = abs(jugador1['ranking'] - jugador2['ranking'])
        
        if (jugador1['elo'] == 1000 or jugador2['elo'] == 1000 or 
            jugador1['ranking'] > 550 or jugador2['ranking'] > 550):
            if total_h2h > 0:   
                peso_base = 0.15
                peso_forma = 0.18
                peso_cuotas = 0.6
                peso_h2h = 0.07
            else:   
                peso_base = 0.4
                peso_forma = 0.1
                peso_cuotas = 0.4
                
        elif (diff_elo >= 115 or diff_ranking >= 25):
            peso_base = 0.45 if total_h2h > 0 else 0.45
            peso_forma = 0.25 if total_h2h > 0 else 0.35
            peso_cuotas = 0.15 if total_h2h > 0 else 0.2
        else:
            peso_base = 0.3 if total_h2h > 0 else 0.35
            peso_forma = 0.4 if total_h2h > 0 else 0.45
            peso_cuotas = 0.15 if total_h2h > 0 else 0.2
            peso_h2h = 0.15 if total_h2h > 0 else 0

        # Ajustes específicos para Challengers
        penalizacion_forma1 = 0
        if es_challenger:
            if (jugador1['elo'] >= 1500 and jugador2['elo'] <= 1499) or (jugador2['elo'] >= 1500 and jugador1['elo'] <= 1499): 
                peso_base = 0.07   
                peso_forma = 0.08   
                peso_cuotas = 0.65 if total_h2h >= 2 else 0.85  
                peso_h2h = 0.2 if total_h2h >= 2 else 0

            elif (jugador1['ranking'] <= 95 and jugador2['ranking'] >= 145) or (jugador2['ranking'] <= 95 and jugador1['ranking'] >= 145): 
                peso_base = 0.07   
                peso_forma = 0.08   
                peso_cuotas = 0.65 if total_h2h >= 2 else 0.85  
                peso_h2h = 0.2 if total_h2h >= 2 else 0

            else:
                peso_base = 0.5  
                peso_forma = 0.4 if total_h2h >= 3 else 0.3
                peso_cuotas = 0.25 if total_h2h >= 3 else 0.55
                peso_h2h = 0.1 if total_h2h >= 2 else 0
            
            if forma1 < forma2:
                penalizacion_forma1 = 0.25   
            else:
                penalizacion_forma1 = 0
        
        # Ajustes específicos para ATP
        if es_atp:
            if (jugador1['ranking'] <= 100 and jugador2['ranking'] >= 165) or (jugador2['ranking'] <= 100 and jugador1['ranking'] >= 165): 
                peso_base = 0.35   
                peso_forma = 0.1   
                peso_cuotas = 0.55   

        # Cálculo de puntuaciones ponderadas
        puntuacion1 = calcular_puntuacion_ponderada(jugador1, cuota1, superficie, es_challenger)
        puntuacion2 = calcular_puntuacion_ponderada(jugador2, cuota2, superficie, es_challenger)
        mostrar_analisis_formula(jugador1, jugador2, puntuacion1, puntuacion2)
           

        # Cálculo de probabilidad final
        probabilidad1 = (peso_base * prob_base1 + 
                        peso_forma * prob_forma1 + 
                        peso_cuotas * prob_cuota1 + 
                        (peso_h2h * prob_h2h1 if total_h2h >= 2 else 0)) - penalizacion_forma1
        
        probabilidad2 = 1 - probabilidad1

        # Ajuste por especialista
        impacto_especialista = calcular_impacto_especialista(jugador1, jugador2)
        probabilidad1 += impacto_especialista
        probabilidad1 = max(0, min(probabilidad1, 0.975))  
        probabilidad2 = 1 - probabilidad1
        
        # Cálculo de EV
        ev1 = (probabilidad1 * cuota1) - 1
        ev2 = (probabilidad2 * cuota2) - 1

        if jugador1.get('no_aplica_sets', False) or jugador2.get('no_aplica_sets', False):
            ev1, ev2 = manejar_no_aplica(jugador1, jugador2, cuota1, cuota2, ev1, ev2)
                  
        # Pesos definidos (Puntuación: 0.65, Emocional: 0.2, EV: 0.15)
        
        if es_challenger:  
            WEIGHTS = {
            'puntuacion': 0.4,
            'emocional': 0.3, 
            'ev': 0.3
            }
        # Índice definitivo (combina componentes con pesos)
            indice_j1 = (
            WEIGHTS['puntuacion'] * puntuacion1 +
            WEIGHTS['emocional'] * indice_emocional1 +
            WEIGHTS['ev'] * max(ev1, 0)  
            )
            indice_j2 = (
            WEIGHTS['puntuacion'] * puntuacion2 +
            WEIGHTS['emocional'] * indice_emocional2 +
            WEIGHTS['ev'] * max(ev2, 0)
            )

        else:    
            WEIGHTS = {
            'puntuacion': 0.6,
            'ev': 0.4
            }

            # Índice definitivo (combina componentes con pesos)
            indice_j1 = (
            WEIGHTS['puntuacion'] * puntuacion1 +
            WEIGHTS['ev'] * max(ev1, 0)  
            )

            indice_j2 = (
            WEIGHTS['puntuacion'] * puntuacion2 +
            WEIGHTS['ev'] * max(ev2, 0)
            )

        # Calcular bonificaciones estratégicas
        bonificacion_j1 = calcular_bonificacion_estrategica(
        jugador1, jugador2, superficie, cuota1, 
        h2h1 > h2h2, jugador1['es_local'], ev1
        )
    
        bonificacion_j2 = calcular_bonificacion_estrategica(
        jugador2, jugador1, superficie, cuota2, 
        h2h2 > h2h1, jugador2['es_local'], ev2
        )

        # Aplicar bonificaciones a los índices
        indice_j1 += bonificacion_j1
        indice_j2 += bonificacion_j2
   
        # Versión optimizada (solo aplica penalización si es necesario)
        if cuota2 > 2.74:
            indice_j2 = indice_j2 * 0.6
            print(f"⚠️ Penalización aplicada a {jugador2['nombre']}")
        
        if cuota1 > 2.74:  
            indice_j1 = indice_j1 * 0.6
            print(f"⚠️ Penalización aplicada a {jugador1['nombre']}")
                    
        # Asegurar que los índices no caigan por debajo de un mínimo (ej: 0.1)
        indice_j1 = max(indice_j1, 0.1)
        indice_j1 = min(indice_j1, 1.0)
        indice_j2 = max(indice_j2, 0.1)
        indice_j2 = min(indice_j2, 1.0)

        # Diferencia de índices y ratio de cuotas
        diff_indice = abs(indice_j1 - indice_j2)
        diff_cuotas = abs(cuota1 - cuota2)  
        diff_elo = abs(jugador1['elo'] - jugador2['elo'])

        gano_sin_ceder_j1 = jugador1.get('gano_sin_ceder_sets') or False
        gano_sin_ceder_j2 = jugador2.get('gano_sin_ceder_sets') or False
        
        flagwin_predicho, indice_ganador_predicho = predecir_flagwin(
            jugador1, jugador2, indice_j1, indice_j2, diff_indice,
            cuota1, cuota2, diff_cuotas, diff_elo,
            jugador1['es_especialista'], jugador2['es_especialista'],
            gano_sin_ceder_j1, gano_sin_ceder_j2,
            h2h1, h2h2, jugador1.get('no_aplica_sets', False),
            jugador2.get('no_aplica_sets', False)
            )
                  
        # Mostrar resultados
        mostrar_titulo("resultados")
        
        if es_challenger:
            print(f"\n● ÍNDICE EMOCIONAL (CHALLENGER)")
            print(f"{jugador1['nombre']}: {indice_emocional1:.2f}/1.0")
            print(f"{jugador2['nombre']}: {indice_emocional2:.2f}/1.0")
                
        print(f"\n● PUNTUACIÓN")
        print(f"Puntuación {jugador1['nombre']}: {puntuacion1:.2f}")
        print(f"Puntuación {jugador2['nombre']}: {puntuacion2:.2f}")
                       
        print(f"\n● FAVORITOS ÍNDICES")
        print(f"Índice {jugador1['nombre']}: {indice_j1:.2f}")
        if indice_j1 > indice_j2:
            print(f"{jugador1['nombre']} tiene mejor índice según el modelo original")
        print(f"Índice {jugador2['nombre']}: {indice_j2:.2f}")
        if indice_j2 > indice_j1:
            print(f"{jugador2['nombre']} tiene mejor índice según el modelo original")
                     
        # Mostrar la predicción en los resultados
        print(f"\n● PREDICCIÓN ÁRBOL DE DECISIÓN")
        if flagwin_predicho == 1:
            if indice_j1 > indice_j2:
                print(f"El árbol predice que {jugador1['nombre']} ganará")
            elif indice_j2 > indice_j1:
                print(f"El árbol predice que {jugador2['nombre']} ganará")
        elif flagwin_predicho == 2:
            if cuota1 > cuota2:
                print(f"El árbol predice que {jugador2['nombre']} ganará")
            elif cuota2 > cuota1:
                print(f"El árbol predice que {jugador1['nombre']} ganará")
        else:
            if indice_j1 > indice_j2:
                print(f"El árbol predice que {jugador2['nombre']} ganará")
            else:
                print(f"El árbol predice que {jugador1['nombre']} ganará")
                       
        # Preguntar por nuevo cálculo
        while True:
            continuar = input("\n¿Desea calcular otro partido? (Sí/No): ").strip().lower()
            
            if continuar in ('sí', 'si', 's'):
                datos_ingresados = {
                    'superficie': None,
                    'torneo': None,
                    'tipo_torneo': None,
                    'jugador1': {},
                    'jugador2': {},
                    'h2h1': None,
                    'h2h2': None,
                    'cuota1': None,
                    'cuota2': None
                }
                print("\033[H  ", end='')
                print("Preparando nuevo cálculo...", end='\r')
                break
            elif continuar in ('no', 'n'):
                mostrar_titulo("¡Gracias por usar la calculadora para tenis!")
                return
            else:
                print("Opción no válida. Por favor, ingrese 'Sí' o 'No'")

if __name__ == "__main__":
    chatbot = TennisChatbot()
    chatbot.start_chat()

#if __name__ == "__main__":
#    calcular_ev_tenis()
