
# Algorithmic Trading Portfolio Optimization

This project implements an automated asset management system, focusing on portfolio optimization and risk management. The core of the system revolves around Mean-Variance Optimization (MVO), LASSO regression, and a risk-averse strategy to ensure a balance between maximizing returns and minimizing transaction costs.

## Key Features

- **Mean-Variance Optimization (MVO)**: A classical approach for portfolio optimization, considering the trade-off between portfolio variance (risk) and expected return.
- **LASSO Regression**: A regularization technique used to perform feature selection and improve the stability of regression models, applied here to estimate the expected returns.
- **Risk-Averse Strategy**: A hybrid method combining LASSO regression with a risk-averse version of MVO, where excessive deviation from the initial portfolio allocation is penalized to reduce transaction costs.
- **Short-Selling Allowed**: Unlike traditional MVO, this strategy allows short-selling while applying constraints to ensure the portfolio is fully invested.

## Files

1. **`estimators.py`**  
   Contains the OLS regression implementation used to estimate the expected returns and covariance matrix of the assets based on factor returns.

2. **`LASSO.py`**  
   Implements LASSO regression for estimating expected returns with an explicit constraint on the L1-norm of the coefficients.

3. **`optimization.py`**  
   Contains the main implementation of the Mean-Variance Optimization (MVO) function, including a risk-averse version with constraints and penalties for deviation from the initial portfolio allocation.

4. **`project_function.py`**  
   A high-level function that integrates the various strategies (LASSO, OLS, and MVO) to execute the portfolio optimization process.

5. **`strategies.py`**  
   Defines different portfolio strategies, including an equal-weight strategy and a risk-averse strategy using the combined MVO and LASSO approach.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository-name.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the portfolio optimization strategy, use the following function call:

```python
from project_function import project_function

# Assuming you have periodReturns, periodFactRet, and initial portfolio weights (x0)
x_optimal = project_function(periodReturns, periodFactRet, x0)
```

This will execute the risk-averse portfolio optimization strategy and return the optimal portfolio weights.

## Evaluation

The algorithm was tested on two unseen datasets, achieving:
- **Sharpe Ratio**: 0.22
- **Turnover Ratio**: 0.00

These results outperformed the average Sharpe ratio of **0.135** and turnover ratio of **0.6415** across competing teams.

## Contributing

Feel free to submit a pull request or open an issue if you have suggestions or encounter any problems.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
