import numpy as np
from services.estimators import *
from services.optimization import *
from services.LASSO import *


# this file will produce portfolios as outputs from data - the strategies can be implemented as classes or functions
# if the strategies have parameters then it probably makes sense to define them as a class


def equal_weight(periodReturns):
    """
    computes the equal weight vector as the portfolio
    :param periodReturns:
    :return:x
    """
    T, n = periodReturns.shape
    x = (1 / n) * np.ones([n])
    return x

class risk_averse_LASSO_OLS_MVO:
    """
    Using our Penalised LASSO regression function to select the best factors for the portfolio
        and our risk averse MVO function to allocate the portfolio
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns, x0):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :param x0:

        :return: x

        """
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q, T = LASSO(returns, factRet, 0.01) 
        x = MVO_Short_Sell_risk_averse(mu, Q, T, x0) 
        return x 
    
# class ROBUST_LASSO_OLS_MVO:
#     """
#     LASSO to select best factors for certain constraint
#     """

#     def __init__(self, NumObs=36):
#         self.NumObs = NumObs  # number of observations to use

#     def execute_strategy(self, periodReturns, factorReturns, x0):
#         """
#         executes the portfolio allocation strategy based on the parameters in the __init__

#         :param factorReturns:
#         :param periodReturns:
#         :return:x
#         """
#         returns = periodReturns.iloc[(-1) * self.NumObs:, :]
#         factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
#         mu, Q, T = LASSO(returns, factRet, 0.01) # here 3 is the constaint on the L1 norm of beta, which means that the number of factors selected will be 3
#         # mu, Q = opt_param_est(B, factRet, returns)
#         x = MVO_Short_Sell_ROBUST(mu, Q, T, x0)
#         return x 
    
class LASSO_OLS_MVO:
    """
    LASSO to select best certain factors instead of all paramaters to reduce overfitting.
    Reduction of overfitting is important in finance as it can lead to poor performance in the future.

    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns, x0 = None):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        # mu, Q = lasso(returns, factRet, 3) # here 3 is the constaint on the L1 norm of beta, which means that the number of factors selected will be 3
        mu, Q, T = LASSO(returns, factRet, 0.01) # here 0.01 is the alpha value for the LASSO regression
        x = MVO_Short_Sell(mu, Q)
        return x 


class HistoricalMeanVarianceOptimization:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns=None):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param periodReturns:
        :param factorReturns:
        :return: x
        """
        factorReturns = None  # we are not using the factor returns
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        print(len(returns))
        mu = np.expand_dims(returns.mean(axis=0).values, axis=1)
        Q = returns.cov().values
        x = MVO(mu, Q)

        return x


class OLS_MVO:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = MVO(mu, Q)
        return x
