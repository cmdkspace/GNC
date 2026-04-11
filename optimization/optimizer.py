# run_optimizer, gradient_descent, finite_diff_gradient

# *** this is for offline trajectory optimization***

from scipy.optimize import minimize
from optimization.cost_function import cost_function

def run_optimizer(theta0, bounds, params, sim_params, gains, q_ref):
    def J(theta):
        return cost_function(theta, params, sim_params, gains, q_ref)
    result = minimize(
        fun = J,
        x0 = theta0, 
        method='L-BFGS-B', 
        bounds=bounds, 
        options={'maxiter' : 50}
    )

    return result.x, result.fun, result