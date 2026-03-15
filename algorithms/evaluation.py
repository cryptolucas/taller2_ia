from __future__ import annotations
from typing import TYPE_CHECKING
from algorithms.utils import bfs_distance

if TYPE_CHECKING:
    from world.game_state import GameState

def evaluation_function(state: GameState) -> float:
    """
    Función de evaluación para estados no terminales del juego Dron vs. Cazadores.
    Combina progreso de entregas, distancia a objetivos y evasión de amenazas.
    """
    # 1. Obtener información básica del estado
    pos_dron = state.get_drone_position()
    pos_cazadores = state.get_hunters_positions()
    entregas_pendientes = state.get_pending_deliveries()
    layout = state.get_layout()
    puntuacion_actual = state.get_score()

    # 2. Inicializar el puntaje con el score actual del juego 
    # El score ya incluye penalizaciones por tiempo/pasos y bonos por entregas.
    evaluacion = puntuacion_actual

    # 3. Penalizar por número de entregas pendientes 
    evaluacion -= 100 * len(entregas_pendientes)

    # 4. Distancia al punto de entrega más cercano 
    if entregas_pendientes:
        distancias_entregas = []
        for entrega in entregas_pendientes:
            dist = bfs_distance(layout, pos_dron, entrega)
            distancias_entregas.append(dist)
        
        # Restamos la distancia mínima: estar más cerca de una entrega aumenta el puntaje
        evaluacion -= min(distancias_entregas)

    # 5. Evasión de Cazadores 
    # Los cazadores solo pueden caminar por espacio libre ('.' / ' ')
    for pos_c in pos_cazadores:
        # Usamos hunter_restricted=True porque los cazadores tienen restricciones de terreno.
        dist_amenaza = bfs_distance(layout, pos_c, pos_dron, hunter_restricted=True)
        
        # Penalización exponencial si el cazador está muy cerca
        if dist_amenaza <= 2:
            evaluacion -= 500  # Peligro crítico
        elif dist_amenaza <= 4:
            evaluacion -= 100  # Precaución
            
    # 6. Normalizar el retorno dentro del rango [-1000, 1000] 
    return max(min(evaluacion, 1000.0), -1000.0)
