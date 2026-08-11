import yfinance as yf
import pandas as pd
import numpy as np
import math


##INPUTS##

S0 = 740.0
K = 750.0
r = 0.0411
q = 0.0102
vol = 0.1511
T= 79/365
N=100


print("Stock:", S0)
print("Strike:", K)
print("Time in days:", T)
print("Steps:", N)

#Model Parameters

dt = T/N
u = math.exp(vol*math.sqrt(dt))
d = 1/u
p = (math.exp((r-q)*dt)-d)/(u-d)
p_down = 1-p
discount_fac = math.exp(-r*dt)

print(f"dt       = {dt:.8f}")
print(f"u        = {u:.8f}")
print(f"d        = {d:.8f}")
print(f"p        = {p:.8f}")
print(f"p_down = {p_down:.8f}")
print(f"discount = {discount_fac:.8f}")

if not 0 <= p <= 1:
    raise ValueError("Invalid risk-neutral probability")
print("Risk-neutral probability check: PASS")


#Model#

Stock_tree = []
for i in range(N+1):
    level = []
    for j in range(i+1):
        stock_price = (S0*(u**j)*(d**(i-j)))
        level.append(stock_price)
    Stock_tree.append(level)

## Terminal Payoffs ##

call_terminal = []
for n in Stock_tree[N]:
    payoff = max(n-K,0)
    call_terminal.append(payoff)


put_terminal = []
for n in Stock_tree[N]:
    payoff = max (K-n,0)
    put_terminal.append(payoff)


#European Call/Put

def european_backward_induction(terminal_values):

    values = terminal_values.copy()

    for i in range(N - 1, -1, -1):
        new_values = []

        for j in range(i + 1):
            down_value = values[j]
            up_value = values[j + 1]

            option_value = discount_fac * ((1 - p) * down_value + p * up_value)
            new_values.append(option_value)

        values = new_values

    return values[0]

european_call = european_backward_induction(call_terminal)
european_put = european_backward_induction(put_terminal)


## American Call/Put

def american_backward_induction(terminal_values,option_type):

    values = terminal_values.copy()

    for i in range(N - 1, -1, -1):
        new_values = []

        for j in range(i + 1):

            # 1. Continuation value
            down_value = values[j]
            up_value = values[j + 1]
            continuation = discount_fac * ((1 - p) * down_value + p * up_value)

            # 2. Stock price at current node
            stock_price = Stock_tree[i][j]

            # 3. Immediate exercise value
            if option_type == "call":
                exercise = max(stock_price - K,0)
            elif option_type == "put":
                exercise = max(K - stock_price, 0)
            else:
                raise ValueError( "option_type must be 'call' or 'put'")

            # 4. American holder chooses best value
            option_value = max(continuation,exercise)
            new_values.append(option_value)

        values = new_values

    return values[0]

american_call = (american_backward_induction(call_terminal,"call"))
american_put = (american_backward_induction(put_terminal, "put"))


call_early_exercise_premium = (american_call - european_call)
put_early_exercise_premium = ( american_put - european_put)

#Dashboard#

print("\n5-Step CRR Model")
print("-" * 35)
print(f"European Call : ${european_call:.4f}")
print(f"American Call : ${american_call:.4f}")
print(f"European Put  : ${european_put:.4f}")
print(f"American Put  : ${american_put:.4f}")
print()
print(f"Call Early Exercise Premium: "f"${call_early_exercise_premium:.4f}")
print(f"Put Early Exercise Premium : "f"${put_early_exercise_premium:.4f}")