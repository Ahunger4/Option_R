

import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义计算Black-Scholes Greeks的函数
def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes Greeks: Delta, Gamma, Theta, Vega, Rho
    S: spot price of the asset
    K: strike price
    T: time to maturity (in years)
    r: risk-free rate
    sigma: volatility of the asset
    option_type: 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = si.norm.cdf(d1)
        theta = -(S*sigma*np.exp(-d1**2/2))/(2*np.sqrt(2*np.pi*T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    else:
        delta = -si.norm.cdf(-d1)
        theta = -(S*sigma*np.exp(-d1**2/2))/(2*np.sqrt(2*np.pi*T)) - r*K*np.exp(-r*T)*norm.cdf(d2) + r*K*np.exp(-r*T)
        
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * si.norm.cdf(d2) if option_type == 'call' else -K * T * np.exp(-r * T) * si.norm.cdf(-d2)
    
    return delta, gamma, theta, vega, rho

# 参数设置
SPOT = 100  
S = np.linspace(80, 120, 100)  
K_put = SPOT * 0.98  
K_call = SPOT * 1.02  
T = 61 / 242  
r_put = 0.025  
r_call = 0.035  
sigma_put = 0.202  
sigma_call = 0.195  

# 计算希腊字母
greeks_put = np.array([bs_greeks(s, K_put, T, r_put, sigma_put, 'put') for s in S])
greeks_call = np.array([bs_greeks(s, K_call, T, r_call, sigma_call, 'call') for s in S])

# 绘图
fig, ax = plt.subplots(5, 1, figsize=(10, 20))
greeks_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']

for i in range(5):
    ax[i].plot(S, greeks_put[:, i] + greeks_call[:, i], label='Total ' + greeks_names[i])
    ax[i].plot(S, greeks_put[:, i], label='Put ' + greeks_names[i], linestyle='--')
    ax[i].plot(S, greeks_call[:, i], label='Call ' + greeks_names[i], linestyle='--')
    ax[i].set_title(greeks_names[i] + ' vs Asset Price')
    ax[i].legend()
    ax[i].set_xlabel('Asset Price')
    ax[i].set_ylabel(greeks_names[i])

plt.tight_layout()
plt.show()


def option_price(S, K, T, r, sigma, q, option_type='call'):
    """
    计算期权价值，考虑隐含贴水q
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * np.exp(-q * T) * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * np.exp(-q * T) * si.norm.cdf(-d1)
    return price

# 重新计算Rho-Q，使用各自的r和sigma
def calculate_rho_q(S, K, T, r, sigma, q, option_type='call', dq=0.00001):
    """
    通过微调隐含贴水q来近似计算Rho-Q
    """
    price_original = option_price(S, K, T, r, sigma, q, option_type)
    price_new = option_price(S, K, T, r, sigma, q + dq, option_type)
    rho_q = (price_new - price_original) / dq
    return rho_q

q = 0.037  # 隐含贴水

# 计算看涨和看跌期权的Rho-Q
rho_q_call = np.array([calculate_rho_q(s, K_call, T, r_call, sigma_call, q, 'call') for s in S])
rho_q_put = np.array([calculate_rho_q(s, K_put, T, r_put, sigma_put, q, 'put') for s in S])
rho_q_total = rho_q_call + rho_q_put

# 绘制Rho-Q图像
plt.figure(figsize=(10, 6))
plt.plot(S, rho_q_call, label='Call Rho-Q')
plt.plot(S, rho_q_put, label='Put Rho-Q')
plt.plot(S, rho_q_total, label='Total Rho-Q', linestyle='--')
plt.title('Rho-Q vs. Asset Price')
plt.xlabel('Asset Price')
plt.ylabel('Rho-Q')
plt.legend()
plt.show()