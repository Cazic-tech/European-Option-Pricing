import yfinance as yf
import pandas as pd
import math
import numpy as np
from datetime import datetime,date
from scipy.stats import norm

#INPUTS

S = 740
X = 750
r = 0.0411
q = 0.0102
vol = 0.1511
expiration = "2026-10-24"
T = (datetime.strptime(expiration, '%Y-%m-%d').date()- date.today()).days/365
discount_fac = math.exp(-r*T)
contract_multiplier = 100

print("-"*30)
print("BLACK-SCHOLES MODEL")
print("-"*30)
print("Valuation day:",date.today())

# Manual Model

d1 = (math.log(S/X)+((r-q)+(vol**2/2))*T)/(vol*math.sqrt(T))
d2 = d1-vol*math.sqrt(T)
N_d1 = norm.cdf(d1)
N_d2 = norm.cdf(d2)
N_d1_inv = norm.cdf(-d1)
N_d2_inv = norm.cdf(-d2)

call_price = S*math.exp((-q)*T)*N_d1-X*discount_fac*N_d2
put_price = X*discount_fac*N_d2_inv-S*math.exp((-q)*T)*N_d1_inv

#Function

def bsm_pricing(stock_price,strike,interest_rate,dividend_yield,volatility,time_in_years):

    S = stock_price
    X =  strike
    r = interest_rate
    q = dividend_yield
    vol = volatility
    T = time_in_years

    d1 = (math.log(S/X)+((r-q)+(vol**2/2))*T)/(vol*math.sqrt(T))
    d2 = d1-vol*math.sqrt(T)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_d1_inv = norm.cdf(-d1)
    N_d2_inv = norm.cdf(-d2)
    call_price = S*math.exp((-q)*T)*N_d1-X*math.exp(-r*T)*N_d2
    put_price = X*math.exp(-r*T)*N_d2_inv-S*math.exp((-q)*T)*N_d1_inv

    return call_price, put_price

call = bsm_pricing(S,X,r,q,vol,T)[0]
put = bsm_pricing(S,X,r,q,vol,T)[1]


print()
print("-"*30)
print("Call:", f"{call:.4}")
print("Call Contract Value: $",f"{call*contract_multiplier:.8}")
print("Put:", f"{put:.4}")
print("Put Contract Value: $",f"{put*contract_multiplier:.8}")

## Measurements and Probaibilities

call_intrinsic = max(S-X,0)
call_extrinsic = call - call_intrinsic
put_intrinsic = max(X-S,0)
call_extrinsic = put - put_intrinsic

call_prop_ITM = N_d2
put_prop_ITM = N_d2_inv

call_breakeven = call + X
put_breakeven = X - put

call_prob_profit = norm.cdf((math.log(S/call_breakeven)+(r-q-(vol**2/2))*T)/(vol*math.sqrt(T)))
put_prob_profit = norm.cdf(-(math.log(S/put_breakeven)+(r-q-(vol**2/2))*T)/(vol*math.sqrt(T)))

print()
print("-"*30)
print("Model Probabilities and Checks")
print("-"*30)
print("Call ITM Probability:",f"{call_prop_ITM:.2%}")
print("Call Breakeven:",f"{call_breakeven:.6}")
print("Call Profit Probability:",f"{call_prob_profit:.2%}")
print()
print("Put ITM Probability:",f"{put_prop_ITM:.2%}")
print("Put Breakeven:",f"{put_breakeven:.6}")
print("Put Profit Probability:",f"{put_prob_profit:.2%}")
print()

###Parity check

left = call - put
right = S*math.exp(-q*T)-X*discount_fac

if abs(right-left)<=0.00001:
    print("Put-Call Parity HOLDS")
else:
    print("Model has an ERROR")

##GREEKS

#---Delta
delta_call = math.exp(-q*T)*N_d1
delta_put = math.exp(-q*T)*N_d1_inv

#---Gamma - same for puts and calls
gamma = (math.exp(-q*T)*norm.pdf(d1))/(S*vol*math.sqrt(T))

#---Vega - same for puts and calls
Vega = math.exp(-q*T)*norm.pdf(d1)*S*math.sqrt(T)/100

#---Rho
rho_call = T*discount_fac*X*N_d2/100
rho_put = -T*discount_fac*X*N_d2_inv/100

#---Phi
phi_call = -(T*S*math.exp(-q*T)*N_d1)/100
phi_put = (T*S*math.exp(-q*T)*N_d1_inv)/100

#--- Theta
ft = (S*math.exp(-q*T)*norm.pdf(d1)*vol)/(2*math.sqrt(T))
theta_call = (-ft-r*X*discount_fac*N_d2+q*S*math.exp(-q*T)*N_d1)/365
theta_put = (-ft+r*X*discount_fac*N_d2_inv-q*S*math.exp(-q*T)*N_d1_inv)/365

print()
print("-"*30)
print("GREEKS")
print("-"*30)
print()
print("CALL:")
print("Delta:",f"{delta_call:.8}")
print("Gamma:",f"{gamma:.8}")
print("Vega:",f"{Vega:.8}")
print("Theta:",f"{theta_call:.8}")
print("Rho:",f"{rho_call:.8}")
print("Phi:",f"{phi_call:.8}")
print()
print("PUT:")
print("Delta:",f"{delta_put:.8}")
print("Gamma:",f"{gamma:.8}")
print("Vega:",f"{Vega:.8}")
print("Theta:",f"{theta_put:.8}")
print("Rho:",f"{rho_put:.8}")
print("Phi:",f"{phi_put:.8}")


## ELASTICITY

call_e = delta_call*(S/call)
put_e = abs(delta_put)*(S/put)


print()
print("-"*30)
print("LEVERAGE")
print("-"*30)
print()
print("Call Leverage:",f"{call_e:.4}x")
print("Put Leverage:",f"{put_e:.4}x")
print()
print("**END OF THE MODEL**")