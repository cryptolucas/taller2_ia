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
    pos_dron = state.get_drone_position()
    pos_cazadores = state.get_hunters_positions()
    entregas_pendientes = state.get_pending_deliveries()
    layout = state.get_layout()
    puntuacion_actual = state.get_score()

    evaluacion = puntuacion_actual

    evaluacion -= 100 * len(entregas_pendientes)

    if entregas_pendientes:
        distancias_entregas = []
        for entrega in entregas_pendientes:
            dist = bfs_distance(layout, pos_dron, entrega)
            distancias_entregas.append(dist)
        
        evaluacion -= min(distancias_entregas)

    for pos_c in pos_cazadores:
        dist_amenaza = bfs_distance(layout, pos_c, pos_dron, hunter_restricted=True)
        
        if dist_amenaza <= 2:
            evaluacion -= 500  
        elif dist_amenaza <= 4:
            evaluacion -= 100  
            
    return max(min(evaluacion, 1000.0), -1000.0)
