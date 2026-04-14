import numpy as np


def evaluate_problem(problem, position):
    cost = problem.eval(position)
    optimum = getattr(problem, "optimum", None)
    if optimum is not None:
        cost = cost - optimum
    return cost


def count_evaluations(position):
    shape = getattr(position, "shape", None)
    if shape is None or len(shape) == 0:
        return 1
    return shape[0] if len(shape) >= 2 else 1


def apply_ela_count(config, current_fes, consumed_fes):
    if getattr(config, "count_ela_fes", False):
        return current_fes + consumed_fes
    return current_fes


def reset_cost_history(owner, initial_best):
    owner.log_index = 1
    owner.cost = [initial_best]


def maybe_record_cost(owner, current_fes, current_best):
    if current_fes >= owner.log_index * owner.log_interval:
        owner.log_index += 1
        owner.cost.append(current_best)


def finalize_cost_history(cost_history, final_best, n_logpoint):
    if len(cost_history) >= n_logpoint + 1:
        cost_history[-1] = final_best
    else:
        cost_history.append(final_best)


def safe_cost_scale(costs, minimum=1e-12):
    max_cost = float(np.max(costs))
    return max(max_cost, minimum)
