from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
    #---Minimax recursive function---
        def minimax_value(current_state: GameState, agent_index: int, depth: int) -> float:
            #---------------------------------
            #Caso Base
            #---------------------------------

            # En A y B se devuelve el estado evaluado si se cumple el condicional, sino se sigue expandiendo el árbol
            #A. Revisar si se llego al final o el cazador atrapo al dron o si se excedio el depht
            if current_state.is_win() or current_state.is_lose() or depth == 0:
                return self.evaluation_function(current_state)

            #B.Revisar si todavia hay acciones legales para el agente actual
            num_agents = current_state.get_num_agents()
            legal_actions = current_state.get_legal_actions(agent_index)#
            if not legal_actions:
                return self.evaluation_function(current_state)
            
            #Calcular el siguiente agente y la profundidad para la siguiente llamada recursiva
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth # Se actualiza el deph si el agente siguiente es el dron
            #---------------------------------
            #MAX
            #---------------------------------
            if agent_index == 0:
                # MAX node (drone)
                max_value = float("-inf")
                for action in legal_actions:
                    succ = current_state.generate_successor(agent_index, action)
                    value = minimax_value(succ, next_agent, next_depth)
                    if value > max_value:
                        max_value = value
                return max_value
            #---------------------------------
            #MIN
            #---------------------------------
            else:
                # MIN node (hunter)
                min_value = float("inf")
                for action in legal_actions:
                    succ = current_state.generate_successor(agent_index, action)
                    value = minimax_value(succ, next_agent, next_depth)
                    if value < min_value:
                        min_value = value
                return min_value
    #-------------------------------------------------------------------------------
        # Escoger mejor acción para el nodo raíz MAX (dron) usando minimax
        legal_actions = state.get_legal_actions(self.index)
        if not legal_actions:
            return None

        best_action = None
        best_value = float("-inf")
        num_agents = state.get_num_agents()

        for action in legal_actions:
            succ = state.generate_successor(self.index, action)
            value = minimax_value(succ, (self.index + 1) % num_agents, self.depth)
            if value > best_value or best_action is None:
                best_value = value
                best_action = action

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        def alphabeta_value(
            current_state: GameState,
            agent_index: int,
            depth: int,
            alpha: float,
            beta: float,
        ) -> float:
            #---------------------------------
            #Caso Base
            #---------------------------------

            # En A y B se devuelve el estado evaluado si se cumple el condicional, sino se sigue expandiendo el árbol
            #A. Revisar si se llego al final o el cazador atrapo al dron o si se excedio el depht
            if current_state.is_win() or current_state.is_lose() or depth == 0:
                return self.evaluation_function(current_state)
            
            #B.Revisar si todavia hay acciones legales para el agente actual
            num_agents = current_state.get_num_agents()
            legal_actions = current_state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(current_state)

            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth
            #---------------------------------
            #MAX
            #---------------------------------
            if agent_index == 0:
                # MAX node (drone)
                value = float("-inf")
                for action in legal_actions:
                    succ = current_state.generate_successor(agent_index, action)
                    value = max(value, alphabeta_value(succ, next_agent, next_depth, alpha, beta))
                    alpha = max(alpha, value)
                    #pruning si value > beta, no se evalua el resto de acciones
                    if value > beta:
                        return value
                return value
            #---------------------------------
            #MAX
            #---------------------------------
            else:
                value = float("inf")
                for action in legal_actions:
                    succ = current_state.generate_successor(agent_index, action)
                    value = min(value, alphabeta_value(succ, next_agent, next_depth, alpha, beta))
                    beta = min(beta, value)
                    if value < alpha:
                        # pruning si value < alpha.
                        return value
                return value
    #-------------------------------------------------------------------------------
    # Escoger mejor acción para el nodo raíz MAX (dron) usando minimax
        legal_actions = state.get_legal_actions(self.index)
        if not legal_actions:
            return None

        best_action = None
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        num_agents = state.get_num_agents()

        for action in legal_actions:
            succ = state.generate_successor(self.index, action)
            value = alphabeta_value(succ, (self.index + 1) % num_agents, self.depth, alpha, beta)
            if value > best_value or best_action is None:
                best_value = value
                best_action = action
    # se actualiza alpha para el nodo raíz MAX después de evaluar cada acción
            alpha = max(alpha, value)
            if best_value > beta:
                # se hace pruning. No se evaulua el resto
                break

        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):


    def get_action(self, state):
        legal_actions = state.get_legal_actions(0) 
        
        best_action = None
        max_v = float('-inf')
        
        for action in legal_actions:
            successor = state.generate_successor(0, action) 
            
            v = self.value(successor, 1, self.depth) 
            
            if v > max_v:
                max_v = v
                best_action = action
        
        return best_action

    def value(self, estado, indice_agente, profundidad):
        """
        Gestiona la transición entre agentes y verifica estados terminales.
        """
        if estado.is_victory():
            return 1000
        
        if estado.is_lose():
            return -1000

        if profundidad == 0:
            return self.evaluation_function(estado)

        if indice_agente == 0:
            return self.max_value(estado, indice_agente, profundidad)
        
        else:
            return self.exp_value(estado, indice_agente, profundidad)

    def max_value(self, estado, indice_agente, profundidad):
        """
        Lógica para el nodo MAX (el Dron).
        Busca maximizar el valor retornado por los sucesores.
        """
        # 1. Obtener las acciones legales para el dron
        acciones_legales = estado.get_legal_actions(indice_agente)
        
        # Si no hay acciones (caso extraño), evaluar el estado actual
        if not acciones_legales:
            return self.evaluation_function(estado)
            
        valor_maximo = float('-inf')
        
        # 2. Iterar sobre cada acción para encontrar el máximo
        for accion in acciones_legales:
            sucesor = estado.generate_successor(indice_agente, accion)
            
            # Tras mover el dron (agente 0), le toca al primer cazador (agente 1)
            # No se reduce la profundidad aún, pues el turno completo no ha terminado
            valor_sucesor = self.value(sucesor, 1, profundidad)
            
            valor_maximo = max(valor_maximo, valor_sucesor)
            
        return valor_maximo

    def exp_value(self, estado, indice_agente, profundidad):

        acciones_legales = estado.get_legal_actions(indice_agente)
        if not acciones_legales:
            return self.evaluation_function(estado)

        accion_optima = None
        distancia_minima = float('inf')
        pos_dron = estado.get_drone_position()

        for accion in acciones_legales:
            sucesor = estado.generate_successor(indice_agente, accion)
            pos_cazador = sucesor.get_hunter_position(indice_agente)
            distancia = abs(pos_dron[0] - pos_cazador[0]) + abs(pos_dron[1] - pos_cazador[1])
            
            if distancia < distancia_minima:
                distancia_minima = distancia
                accion_optima = accion

        p = self.p  
        num_acciones = len(acciones_legales)
        prob_aleatoria = p / num_acciones
        
        valor_esperado = 0
        
        proximo_agente = indice_agente + 1
        proxima_profundidad = profundidad
        
        if proximo_agente >= estado.get_num_agents():
            proximo_agente = 0
            proxima_profundidad = profundidad - 1

        for accion in acciones_legales:
            sucesor = estado.generate_successor(indice_agente, accion)
            valor_sucesor = self.value(sucesor, proximo_agente, proxima_profundidad)
            
            probabilidad = prob_aleatoria
            if accion == accion_optima:
                probabilidad += (1 - p)
            
            valor_esperado += probabilidad * valor_sucesor
            
        return valor_esperado