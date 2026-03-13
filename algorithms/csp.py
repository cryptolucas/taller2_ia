from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    
    # Diccionario para rastrear el rendimiento
    stats = {
        "assignments": 0,
        "backtracks": 0
    }

    def recursive_backtracking(csp, assignment):
        if csp.is_complete(assignment):
            return assignment
        
        unassigned_vars = csp.get_unassigned_variables(assignment)
        var = unassigned_vars[0]
        
        for choice in csp.domains[var]:
            
            stats["assignments"] += 1
            
            if csp.is_consistent(var, choice, assignment):
                csp.assign(var, choice, assignment)
                
                result = recursive_backtracking(csp, assignment)
                if result is not None:
                    return result
                
                # Log: Si llegamos aquí, la asignación no funcionó. Backtrack.
                stats["backtracks"] += 1
                csp.unassign(var, assignment)
            else:
                # Opcional: Contar como backtrack si la consistencia falla de inmediato
                stats["backtracks"] += 1
          
        return None
    
    assignment = {}
    final_result = recursive_backtracking(csp, assignment)
    
    # Imprimir resumen de métricas
    print("--- Resumen de Búsqueda ---")
    print(f"Asignaciones intentadas: {stats['assignments']}")
    print(f"Retrocesos (Backtracks): {stats['backtracks']}")
    print("---------------------------")
    
    return final_result


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    stats = {
        "assignments": 0,
        "backtracks": 0,
        "pruned_branches": 0  # Nueva métrica: ramas cortadas por FC
    }

    def forward_checking_backtracking(csp, assignment):
        if csp.is_complete(assignment):
            return assignment
        
        unassigned_vars = csp.get_unassigned_variables(assignment)
        if not unassigned_vars:
            return None
        var = unassigned_vars[0]
        
        for value in csp.domains[var]:
        
            stats["assignments"] += 1
            
            if csp.is_consistent(var, value, assignment):
              
                csp.assign(var, value, assignment)
                old_domains = copy.deepcopy(csp.domains)
                
                is_valid_path = True
                unassigned_neighbors = [v for v in csp.get_neighbors(var) if v not in assignment]
                
                for neighbor in unassigned_neighbors:
                    new_domain = [val for val in csp.domains[neighbor] 
                                 if csp.is_consistent(neighbor, val, assignment)]
                    csp.domains[neighbor] = new_domain
                    
                    if not new_domain:
                        is_valid_path = False
                        stats["pruned_branches"] += 1 
                        break
                
                if is_valid_path:
                    result = forward_checking_backtracking(csp, assignment)
                    if result is not None:
                        return result
                
                
                stats["backtracks"] += 1
                csp.domains = old_domains
                csp.unassign(var, assignment)
            else:
                stats["backtracks"] += 1
                
        return None

    assignment = {}
    final_result = forward_checking_backtracking(csp, assignment)

    print("--- Resumen Forward Checking ---")
    print(f"Asignaciones intentadas: {stats['assignments']}")
    print(f"Ramas podadas por FC: {stats['pruned_branches']}")
    print(f"Retrocesos (Backtracks): {stats['backtracks']}")
    print("--------------------------------")
    
    return final_result


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    # TODO: Implement your code here
    return None


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    # TODO: Implement your code here (BONUS)
    return None
