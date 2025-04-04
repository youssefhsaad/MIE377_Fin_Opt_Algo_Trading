import cvxpy as cp
import numpy as np
# from scipy.stats import chi2


def MVO(mu, Q):
    """
    #---------------------------------------------------------------------- Use this function to construct an example of a MVO portfolio.
    #
    # An example of an MVO implementation is given below. You can use this
    # version of MVO if you like, but feel free to modify this code as much
    # as you need to. You can also change the inputs and outputs to suit
    # your needs.

    # You may use quadprog, Gurobi, or any other optimizer you are familiar
    # with. Just be sure to include comments in your code.

    # *************** WRITE YOUR CODE HERE ***************
    #----------------------------------------------------------------------
    """

    # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb])
    prob.solve(verbose=False)
    return x.value

def MVO_Short_Sell(mu, Q):
    """
    ---------------------------------------------------------------------- 
    This function utilizes MVO to construct a portfolio that allows for short selling.
    The function uses the average of all expected returns as the target return.

    Inputs:
    mu: A numpy array of expected returns for each asset
    Q: A numpy array of the covariance matrix of the returns of each asset
    x0: A numpy array of the initial portfolio weights

    Returns:
    x.value: A numpy array of the optimal portfolio weights
    ----------------------------------------------------------------------
    """

    # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Turnover constraint
    to = np.ones(n)
    to *= 0.2

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                      [A @ x <= b,
                       Aeq @ x == beq])
    prob.solve(verbose=False)

    return x.value

def MVO_Short_Sell_risk_averse(mu, Q, T, x0):
    """
    -----------------------------------------------------------------------
    This is our final version of the MVO function. It allows short selling and
    includes the risk-averse variance formulation as we do not have a set return 
    target. We also included a penalty on deviation from the initial portfolio 
    to prevent excessive transactions.

    Inputs:
    - mu: numpy array of expected returns for each asset
    - Q: numpy array of covariance matrix
    - T: integer representing the number of time periods
    - x0: numpy array of initial portfolio weights

    Returns:
    - x.value: numpy array of optimal portfolio weights
    -----------------------------------------------------------------------
    """
    # Number of assets
    n = len(mu)
    
    # Hyperparameters from robust formulation
    lambda_val = 14              # risk aversion parameter on Q
    gamma_reg = 0.4            # penalty on deviation from initial portfolio (L1 regularization)
    
    # Define the optimization variable
    x = cp.Variable(n)
    
    # Build  objective function:
    # 1. lambda_val * x'Qx penalizes variance.
    # 2. - mu' x rewards expected return.
    # 3. gamma_reg * norm(x - x0, 1) penalizes excessive deviations from the initial portfolio.
    objective = cp.Minimize(
        lambda_val * cp.quad_form(x, Q) 
        - mu.T @ x
        + gamma_reg * cp.norm(x - x0, 1))
    
    # Define constraints:
    constraints = [cp.sum(x) == 1]            # fully invested portfolio
    
    # Form and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    return x.value

# def MVO_Short_Sell_ROBUST(mu, Q, T, x0):
#     # Number of assets
#     n = len(mu)
    
#     # Hyperparameters from robust formulation
#     lambda_val = 14              # risk aversion parameter on Q
#     alpha = 0.7                 # significance level for uncertainty
#     epsilon = np.sqrt(chi2.ppf(alpha, n))

    
#     # Calculate theta as element-wise scaling vector based on Q
#     theta_diag = np.sqrt(np.diag(Q) / T)
    
#     # Additional robust hyperparameters:
#     gamma_reg = 0.4            # penalty on deviation from initial portfolio (L1 regularization)
    
#     # Define the optimization variable
#     x = cp.Variable(n)
    
#     # Build robust objective function:
#     # 1. lambda_val * x'Qx penalizes variance.
#     # 2. - mu' x rewards expected return.
#     # 3. epsilon * norm(theta .* x, 2) robustifies against uncertainty in risk.
#     # 4. gamma_reg * norm(x - x0, 1) penalizes excessive deviations from the initial portfolio.
#     objective = cp.Minimize(
#         lambda_val * cp.quad_form(x, Q) 
#         # cp.quad_form(x, Q) 
#         - mu.T @ x
#         + epsilon * cp.norm(cp.multiply(theta_diag, x), 2)
#         + gamma_reg * cp.norm(x - x0, 1)
#     )
    
#     # Define constraints:
#     constraints = [cp.sum(x) == 1]            # fully invested portfolio
    
#     # Form and solve the problem
#     prob = cp.Problem(objective, constraints)
#     prob.solve(verbose=False)
    
#     # print("Optimal value:", prob.value)
#     # print("Optimal weights:", x.value)
#     print("Alpha:", alpha)
#     return x.value