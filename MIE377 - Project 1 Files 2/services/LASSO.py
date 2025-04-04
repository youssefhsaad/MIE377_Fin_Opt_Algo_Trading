import numpy as np
import cvxpy as cp
from sklearn.linear_model import Lasso


def lasso(returns, factRet, C):
    """
    Solve LASSO regression with an explicit constraint on the L1 norm of beta.

    Returns:
    - mu: Expected returns
    - Q: Covariance matrix
    """

    [T, n] = returns.shape
    [T, p] = factRet.shape

    B = np.zeros((p+1, n))

    beta = cp.Variable((p+1, n))  # Factor loadings

    # Data Matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)

    # print(f"beta shape: {beta.shape}")
    # print(f"X shape: {X.shape}")
    # print(f"returns shape: {returns.shape}")
    # print(f"Operation shape: {(X @ beta).shape}")

    # Objective: Minimize squared errors
    objective = cp.Minimize(cp.sum_squares(returns.values - X @ beta)) 

    # Constraint: L1-norm of beta <= C
    constraints = [cp.norm1(beta[:, i]) <= C for i in range(n)]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # print(beta.value)

    B = beta.value # Factor loadings

    # Separate B into alpha and betas
    a = B[0, :]
    V = B[1:, :]

    # now we need to determine the mu and Q matrix

    # Residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)

    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values

    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D

    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2

    # print(f'mu: {mu}')
    # print(f'Q: {Q}')

    return mu, Q  # Return optimized factor loadings


def LASSO(returns, factRet, alpha=0.01):
    """
    Perform LASSO regression for portfolio optimization.
    
    Parameters:
    - returns: Asset returns (T x N matrix, where T = time periods, N = assets).
    - factRet: Factor returns (T x p matrix, where p = number of factors).
    - alpha: Regularization parameter (controls the strength of the L1 penalty).
    
    Returns:
    - mu: Expected asset returns (N x 1 vector).
    - Q: Covariance matrix of asset returns (N x N matrix).
    - T: Number of time periods.
    """
    # Number of observations and factors
    T, p = factRet.shape
    
    # Data matrix (add intercept column)
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)
    
    # Initialize variables
    N = returns.shape[1]  # Number of assets
    a = np.zeros(N)       # Alpha (intercept)
    V = np.zeros((p, N))  # Betas (factor loadings)
    
    # Perform LASSO regression for each asset
    for i in range(N):
        y = returns.iloc[:, i]  # Target variable (returns for asset i)
        
        # Fit LASSO model
        lasso = Lasso(alpha=alpha, fit_intercept=False) 
        lasso.fit(X, y)
        
        # Extract coefficients
        B = lasso.coef_
        a[i] = B[0]      # Intercept (alpha)
        V[:, i] = B[1:]  # Factor loadings (betas)

    # Residual variance
    ep = returns - X @ np.vstack([a, V]) 
    sigma_ep = 1 / (T - p - 1) * np.sum(ep ** 2, axis=0)
    D = np.diag(sigma_ep)
    
    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values
    
    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D
    
    # Ensure covariance matrix symmetry
    Q = (Q + Q.T) / 2
    return mu, Q, T