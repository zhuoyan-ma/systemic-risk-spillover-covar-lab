"""Parametric bivariate copulas selected by maximum likelihood and AIC."""

import numpy as np
from scipy import stats
from scipy.optimize import minimize


class CopulaModel:
    """Fit Gaussian and Archimedean copulas to pseudo-observations."""
    
    def __init__(self, u_x, u_y):
        self.u_x = np.clip(np.array(u_x), 1e-10, 1-1e-10)
        self.u_y = np.clip(np.array(u_y), 1e-10, 1-1e-10)
        self.results = {}
        self.best_name = None
        self.best_theta = None
        self.best_aic = None
        
    # Copula densities
    
    def _gaussian_pdf(self, u1, u2, rho):
        if abs(rho) >= 1:
            rho = 0.99 * np.sign(rho)
        x1 = stats.norm.ppf(u1)
        x2 = stats.norm.ppf(u2)
        pdf = (1 / np.sqrt(1 - rho**2)) * np.exp(
            -(rho**2 * (x1**2 + x2**2) - 2*rho*x1*x2) / (2*(1 - rho**2))
        )
        return np.clip(pdf, 1e-15, 1e10)
    
    def _clayton_pdf(self, u1, u2, theta):
        if theta <= 0:
            return np.ones_like(u1)
        a = u1**(-theta) + u2**(-theta) - 1
        pdf = (1 + theta) * (u1 * u2)**(-theta - 1) * a**(-1/theta - 2)
        return np.clip(pdf, 1e-15, 1e10)
    
    def _gumbel_pdf(self, u1, u2, theta):
        if theta <= 1:
            theta = 1.001
        lnu1 = -np.log(u1 + 1e-10)
        lnu2 = -np.log(u2 + 1e-10)
        a = lnu1**theta + lnu2**theta
        log_pdf = -a**(1/theta) + (1/theta - 2)*np.log(a) + \
                  (theta-1)*(np.log(lnu1) + np.log(lnu2)) + \
                  np.log(1 + (theta-1)*a**(-1/theta)) + np.log(1/(u1*u2))
        return np.exp(np.clip(log_pdf, -100, 100))
    
    def _frank_pdf(self, u1, u2, theta):
        if abs(theta) < 1e-6:
            return np.ones_like(u1)
        num = -theta * (np.exp(-theta) - 1) * np.exp(-theta * (u1 + u2))
        den = (np.exp(-theta) - 1 + (np.exp(-theta*u1) - 1) * (np.exp(-theta*u2) - 1))**2
        return np.clip(num / (den + 1e-10), 1e-15, 1e10)
    
    # Maximum-likelihood estimation
    
    def fit_copula(self, copula_type):
        """Estimate one copula family by maximum likelihood."""
        u1, u2 = self.u_x, self.u_y
        
        if copula_type == 'gaussian':
            tau = stats.kendalltau(u1, u2)[0]
            rho_init = np.sin(np.pi * tau / 2)
            def neg_log_lik(rho):
                if abs(rho) >= 1:
                    return 1e10
                return -np.sum(np.log(self._gaussian_pdf(u1, u2, rho) + 1e-10))
            result = minimize(neg_log_lik, rho_init, bounds=[(-0.99, 0.99)], method='L-BFGS-B')
            theta, k = result.x[0], 1
            
        elif copula_type == 'clayton':
            def neg_log_lik(theta):
                if theta <= 0:
                    return 1e10
                return -np.sum(np.log(self._clayton_pdf(u1, u2, theta) + 1e-10))
            result = minimize(neg_log_lik, 1.0, bounds=[(0.001, 50)], method='L-BFGS-B')
            theta, k = result.x[0], 1
            
        elif copula_type == 'gumbel':
            def neg_log_lik(theta):
                if theta <= 1:
                    return 1e10
                return -np.sum(np.log(self._gumbel_pdf(u1, u2, theta) + 1e-10))
            result = minimize(neg_log_lik, 1.5, bounds=[(1.001, 20)], method='L-BFGS-B')
            theta, k = result.x[0], 1
            
        elif copula_type == 'frank':
            def neg_log_lik(theta):
                if abs(theta) > 50:
                    return 1e10
                return -np.sum(np.log(self._frank_pdf(u1, u2, theta) + 1e-10))
            result = minimize(neg_log_lik, 1.0, bounds=[(-50, 50)], method='L-BFGS-B')
            theta, k = result.x[0], 1
            
        else:
            raise ValueError(f"Unsupported copula: {copula_type}")
        
        log_lik = -result.fun
        aic = -2 * log_lik + 2 * k
        
        return {'theta': theta, 'log_lik': log_lik, 'aic': aic, 'k': k}
    
    def fit_all(self, copula_list=None):
        """Fit candidate families and select the model with minimum AIC."""
        if copula_list is None:
            copula_list = ['gaussian', 'clayton', 'gumbel', 'frank']
        
        for cop_type in copula_list:
            try:
                self.results[cop_type] = self.fit_copula(cop_type)
            except (ValueError, FloatingPointError) as exc:
                print(f"    {cop_type} failed: {exc}")
                self.results[cop_type] = {'theta': None, 'log_lik': -1e10, 'aic': 1e10}
        
        # Select the lowest-AIC candidate.
        self.best_name = min(self.results, key=lambda x: self.results[x]['aic'])
        self.best_theta = self.results[self.best_name]['theta']
        self.best_aic = self.results[self.best_name]['aic']
        
        return self.results
    
    # Conditional quantiles
    
    def conditional_quantile(self, u_given, alpha, axis='x'):
        """Solve v such that the conditional copula CDF equals alpha."""
        from scipy.optimize import brentq
        
        if axis not in {'x', 'y'}:
            raise ValueError("axis must be 'x' or 'y'.")
        copula_type = self.best_name
        theta = self.best_theta
        
        def objective(v):
            if v <= 0 or v >= 1:
                return 1e10 - alpha
            
            if copula_type == 'clayton':
                # Partial derivative of the Clayton CDF.
                cond_cdf = u_given**(-theta-1) * (u_given**(-theta) + v**(-theta) - 1)**(-1/theta - 1)
                
            elif copula_type == 'gumbel':
                lnu = -np.log(u_given + 1e-10)
                lnv = -np.log(v + 1e-10)
                a = lnu**theta + lnv**theta
                c = np.exp(-a**(1/theta))
                cond_cdf = c * lnu**(theta-1) / (u_given * a**(1 - 1/theta) + 1e-10)
                
            elif copula_type == 'gaussian':
                rho = theta
                if abs(rho) >= 1:
                    rho = 0.99 * np.sign(rho)
                x1 = stats.norm.ppf(u_given)
                x2 = stats.norm.ppf(v)
                cond_cdf = stats.norm.cdf((x2 - rho * x1) / np.sqrt(1 - rho**2))
                
            else:  # frank
                eps = 1e-6
                u1 = max(u_given - eps, eps)
                u2 = min(u_given + eps, 1-eps)
                # Frank CDF, differentiated numerically.
                def frank_cdf(u, v):
                    if abs(theta) < 1e-6:
                        return u * v
                    return -1/theta * np.log(1 + (np.exp(-theta*u)-1)*(np.exp(-theta*v)-1)/(np.exp(-theta)-1))
                cond_cdf = (frank_cdf(u2, v) - frank_cdf(u1, v)) / (u2 - u1)
            
            return cond_cdf - alpha
        
        try:
            return brentq(objective, 0.0001, 0.9999, maxiter=100)
        except ValueError as exc:
            raise RuntimeError(
                f"Conditional CDF inversion failed for {copula_type}."
            ) from exc
    
    def get_theta(self):
        return self.best_theta
    
    def get_best_name(self):
        return self.best_name
    
    def get_aic(self):
        return self.best_aic
