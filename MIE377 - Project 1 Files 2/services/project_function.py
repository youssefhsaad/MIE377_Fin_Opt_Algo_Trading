from services.strategies import *


def project_function(periodReturns, periodFactRet, x0):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :param x0:
    
    :return: the allocation as a vector
    """
    Strategy = risk_averse_LASSO_OLS_MVO()
    x = Strategy.execute_strategy(periodReturns, periodFactRet, x0)
    return x
