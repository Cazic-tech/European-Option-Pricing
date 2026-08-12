# assumes normal distribution instead of log-normal
# it is more appropriate for assets with small price fluctuations, it also allows for negative assets prices,
# whihc can be used on commodities like oil (2020) to handle negative prices

# Libraries 
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def call_price(sigma, S, K, r, t):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    C = np.multiply(S, norm.cdf(d1)) - np.multiply(norm.cdf(d2) * K, np.exp(-r * t))
    return C

def put_price(sigma, S, K, r, t):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    P = -np.multiply(S, norm.cdf(-d1)) + np.multiply(norm.cdf(-d2) * K, np.exp(-r * t))
    return P


def bachelier(sigma, S, K, r, t):
    d = (S - K) / (sigma * np.sqrt(t))
    C = np.exp(-r * t) * ((S - K) * norm.cdf(d) + sigma * np.sqrt(t) * norm.pdf(d))
    return C

def bachelier_call(sigma_n, S, K, r, t):
    F = S * np.exp(r * t)
    d = (F - K) / (sigma_n * np.sqrt(t))
    C = np.exp(-r * t) * ((F - K) * norm.cdf(d)+ sigma_n * np.sqrt(t) * norm.pdf(d))
    return C

def bachelier_put(sigma_n, S, K, r, t):
    F = S * np.exp(r * t)
    d = (F - K) / (sigma_n * np.sqrt(t))
    P = np.exp(-r * t) * ((K - F) * norm.cdf(-d)+ sigma_n * np.sqrt(t) * norm.pdf(d))
    return P



#Inputs 

t = 1 / 12    #  Time to expiration
sigma = 0.20    #  Black Scholes implied volatility
K = 105         #  Strike Price
r = 0.01 
S0 = 100       #  Risk-free rate
S = np.linspace(90, 110, 100)    # Let our stock price range between $90 and $110
sigma_n = sigma*S0

#  Calculate Black Schole call, put price
C_BlackSholes = call_price(sigma, S, K, r, t)
P_BlackSholes = put_price(sigma, S, K, r, t)

#single values
C_BS_single = call_price(sigma, S0, K, r, t)
P_BS_single = put_price(sigma, S0, K, r, t)

#  Calculate Bachelier call price
C_Bachelier = bachelier(sigma * S, S, K, r, t)
P_Bachelier = C_Bachelier - S + np.exp(-r * t) * K

#single values
C_SBachelier = bachelier(sigma_n, S0, K, r, t)
P_SBachelier = C_SBachelier - S0 + np.exp(-r * t) * K

C_ba = bachelier_call(sigma_n, S0, K, r, t)
P_ba = bachelier_put(sigma_n, S0, K, r, t)

print("-"*30)
print("Model Outputs")
print("-"*30)
print()
print("Black-Scholes")
print("Call: ",f"{C_BS_single:.6}")
print("Put: ",f"{P_BS_single:.6}")
print()
print("Bachelier")
print("Call: ",f"{C_SBachelier:.6}")
print("Put: ",f"{P_SBachelier:.6}")
print()
print("Bachelier with Foward Price")
print("Call: ",f"{C_ba:.6}")
print("Put: ",f"{P_ba:.6}")




plt.figure(figsize=(10,5))
plt.plot(S, C_BlackSholes, 'b-', label = 'Black Scholes')
plt.plot(S, C_Bachelier, 'r.', label = 'Bachelier')
plt.grid(True)
plt.legend()
plt.xlabel('Stock Price ($)')
plt.ylabel('Call Price ($)')
plt.title("Black-Scholes vs Bachelier - Call")
plt.show()



#  Plot the results
plt.figure(figsize=(10,5))
plt.plot(S, P_BlackSholes, 'b-', label = 'Black Scholes')
plt.plot(S, P_Bachelier, 'r.', label = 'Bachelier')
plt.grid(True)
plt.legend()
plt.xlabel('Stock Price ($)')
plt.ylabel('Put Price ($)')
plt.title("Black-Scholes vs Bachelier - Put")
plt.show()

