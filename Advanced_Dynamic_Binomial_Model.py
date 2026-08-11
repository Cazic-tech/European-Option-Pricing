# 1- Libraries

import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib as mpl
from datetime import datetime
from scipy.optimize import brentq

# =========================================================
# 2. USER INPUTS AND MANUAL OVERRIDE SWITCHES
# =========================================================

# --- Market selection ---
symbol = "SPY"
spot_mode = "auto"                   # Allowed: "auto", "manual"
manual_spot = 740.0
expiration_mode = "index"            # Allowed: "index", "manual"
expiration_index = 5
manual_expiration = "2026-10-16"
K_market = 770.0
N_dynamic = 2000

# --- Market data settings ---
recent_price_period = "5d"
historical_vol_period = "5y"
near_spot_range = 0.05
show_available_expirations = False
show_near_spot_options = False

# --- Dividend yield ---
dividend_mode = "auto"               # Allowed: "auto", "manual"
manual_dividend_yield = 0.0102

# --- Risk-free rate ---
rate_mode = "auto"                   # Allowed: "auto", "manual"
manual_risk_free_rate = 0.0411

# --- Valuation volatility ---
volatility_mode = "historical"        # Allowed: "historical", "manual"
manual_volatility = 0.1511

# --- CRR implied volatility solver ---
iv_style = "american"                # Allowed: "european", "american"
iv_sigma_low = 0.01
iv_sigma_high = 3.00

convergence_steps = [5, 25, 50, 100, 250, 500, 1000, 2000]
convergence_tolerance = 0.01       # Maximum desired price difference in dollars

# --- Greeks engine ---
greek_volatility_mode = "model"       # Allowed: "model", "crr_iv"
greek_steps = 2000                    # Use the stable tree size found in convergence          
greek_vol_bump = 0.005                # 0.50 volatility-point bump
greek_rate_bump = 0.001               # 0.10 percentage-point rate bump
greek_dividend_bump = 0.001           # 0.10 percentage-point dividend bump

# =========================================================
# 3. FUNCTIONS
# =========================================================


# ---------------------------------------------------------
# 3.1 CRR OPTION PRICING FUNCTION
# ---------------------------------------------------------


def crr_price(S, K, r, q, sigma, T, N=100, option_type="call", style="european"):
    
    """
    Price European or American options using the Cox-Ross-Rubinstein binomial model.

    Parameters
    ----------
    S : Current stock price.
    K : Strike price.
    r : Annual risk-free rate as decimal.
    q : Annual dividend yield as decimal.
    sigma : Annual volatility as decimal.
    T : Time to expiration in years.
    N : Number of binomial steps.
    option_type : "call" or "put".
    style : "european" or "american".
    """

    # -------------------------
    # 1. Validate inputs
    # -------------------------

    if S <= 0:
        raise ValueError("Stock price must be positive.")

    if K <= 0:
        raise ValueError("Strike price must be positive.")

    if sigma <= 0:
        raise ValueError("Volatility must be positive.")

    if T <= 0:
        raise ValueError("Time to expiration must be positive.")

    if N < 1:
        raise ValueError("N must be at least 1.")

    option_type = option_type.lower()
    style = style.lower()

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    if style not in ("european", "american"):
        raise ValueError("style must be 'european' or 'american'")

    # -------------------------
    # 2. CRR parameters
    # -------------------------

    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = ( math.exp((r - q) * dt) - d) / (u - d)
    discount = math.exp( -r * dt )

    if not 0 <= p <= 1:
        raise ValueError( f"Invalid risk-neutral probability: {p}" )

    # -------------------------
    # 3. Terminal stock prices
    # -------------------------

    j = np.arange(N + 1)
    stock_prices = ( S* u**j * d**(N - j))

    # -------------------------
    # 4. Terminal payoffs
    # -------------------------

    if option_type == "call":
        option_values = np.maximum(stock_prices - K,0.0 )

    else:
        option_values = np.maximum( K - stock_prices,0.0)

    # -------------------------
    # 5. Backward induction
    # -------------------------

    for i in range(N - 1, -1, -1):

        continuation = discount * ((1 - p) * option_values[:-1] + p * option_values[1:])

        if style == "european":
            option_values = continuation

        else:
            j = np.arange(i + 1)

            stock_prices = (S* u**j* d**(i - j))

            if option_type == "call": 
                exercise = np.maximum(stock_prices - K, 0.0)

            else:
                exercise = np.maximum(K - stock_prices,0.0)

            option_values = np.maximum( continuation,exercise)

    # -------------------------
    # 6. Today's price
    # -------------------------

    return float(option_values[0])

# ---------------------------------------------------------
# 3.2 PRICE ALL STYLES HELPER
# ---------------------------------------------------------

def price_all_styles(S, K, r, q, sigma, T, N=100):

    results = { 
        "European Call": crr_price( S, K, r, q, sigma, T, N, "call", "european" ),
        "European Put": crr_price( S, K, r, q, sigma, T,N, "put", "european"),
        "American Call": crr_price( S, K, r, q, sigma, T, N, "call", "american"),
        "American Put": crr_price( S, K, r, q, sigma, T,N, "put", "american")
    }

    return results

# ---------------------------------------------------------
# 3.3 BID/ASK QUOTE QUALITY FUNCTION
# ---------------------------------------------------------

def quote_quality(bid, ask):
    if bid < 0 or ask < 0:
        return {"status": "INVALID", "mid": np.nan, "spread": np.nan, "spread_pct": np.nan}
    if ask < bid:
        return {"status": "CROSSED", "mid": np.nan, "spread": np.nan, "spread_pct": np.nan}
    if bid == 0 and ask == 0:
        return {"status": "NO QUOTE", "mid": 0.0, "spread": 0.0, "spread_pct": np.nan}

    mid = (bid + ask) / 2
    spread = ask - bid
    spread_pct = spread / mid if mid > 0 else np.nan
    status = "NO BID" if bid == 0 else "OK"

    return {"status": status, "mid": mid, "spread": spread, "spread_pct": spread_pct}

# ---------------------------------------------------------
# 3.4 DIVIDEND YIELD FUNCTION
# ---------------------------------------------------------

def get_dividend_yield(ticker, spot):
    info = ticker.info
    dividend_yield = info.get("dividendYield")
    dividend_rate = info.get("dividendRate")
    trailing_yield = info.get("trailingAnnualDividendYield")
    

    if dividend_yield is not None:
        return float(dividend_yield) / 100, "Yahoo dividendYield"
    if dividend_rate is not None and spot > 0:
        return float(dividend_rate) / spot, "Yahoo dividendRate / Spot"
    if trailing_yield is not None:
        return float(trailing_yield), "Yahoo trailingAnnualDividendYield"

    return 0.0, "No dividend data"


# In this section we can choose to use manual inputed date for the dividend yield or 
# automatic data getted from functions and yahoo finance

# ---------------------------------------------------------
# 3.5 TREASURY RISK-FREE RATE FUNCTION
# ---------------------------------------------------------

def get_treasury_rate(days_to_expiration):

    current_year = pd.Timestamp.today().year
    url = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView"
        f"?type=daily_treasury_bill_rates&field_tdr_date_value={current_year}")
    
    tables = pd.read_html(url)
    treasury = tables[0].copy()
    rates = treasury.iloc[:,[0, 5, 7, 9, 11, 13, 15, 17]].copy()
    rates.columns = ["Date",28,42,56,91,119,182, 364]
    rates["Date"] = pd.to_datetime(rates["Date"],errors="coerce")

    # Convert yields to numbers
    for column in rates.columns[1:]:
        rates[column] = pd.to_numeric(rates[column],errors="coerce")

    rates = rates.dropna(subset=["Date"])
    latest = rates.iloc[-1]
    maturities = np.array([28, 42, 56, 91, 119, 182, 364],dtype=float)
    yields = np.array([latest[28],latest[42],latest[56],latest[91],latest[119],latest[182],latest[364]], dtype=float)
    valid = ~np.isnan(yields)
    maturities = maturities[valid]
    yields = yields[valid]
    rate_percent = np.interp(days_to_expiration,maturities,yields)
    rate_decimal = (rate_percent / 100)

    return {"rate": rate_decimal,"rate_percent": rate_percent,
        "date": latest["Date"],"days": days_to_expiration}


# ---------------------------------------------------------
# 3.6 HISTORICAL VOLATILITY FUNCTION
# ---------------------------------------------------------

def get_historical_volatility(ticker,period="1y"):
    prices = ticker.history( period=period, auto_adjust=True)["Close"].dropna()

    if len(prices) < 2:
        raise ValueError("Not enough historical prices.")

    log_returns = np.log(prices / prices.shift(1)).dropna()
    historical_vol = (log_returns.std()* np.sqrt(252))

    return float(historical_vol)

# ---------------------------------------------------------
# 3.7 CRR IMPLIED VOLATILITY FUNCTION
# ---------------------------------------------------------

def crr_implied_volatility(market_price, S, K, r, q, T, N, option_type, style="american", sigma_low=0.01, sigma_high=3.00):
    """Solve for the volatility that makes the CRR model price equal to the observed market price."""

    if market_price <= 0:
        raise ValueError("Market price must be positive.")

    def objective(sigma):
        model_price = crr_price(S=S, K=K, r=r, q=q, sigma=sigma, T=T, N=N, option_type=option_type, style=style)
        return model_price - market_price

    try:
        return float(brentq(objective, sigma_low, sigma_high))
    except ValueError:
        return None

# =========================================================
# 3.8 CRR Convergence Analysis Function
# =========================================================

def crr_convergence_analysis(S, K, r, q, sigma, T, steps):

    records = []

    for N in steps:
        prices = price_all_styles(S=S, K=K, r=r, q=q, sigma=sigma, T=T, N=N)
        records.append({
            "Steps": N,
            "European Call": prices["European Call"],
            "American Call": prices["American Call"],
            "European Put": prices["European Put"],
            "American Put": prices["American Put"]
        })

    convergence = pd.DataFrame(records)
    reference = convergence.iloc[-1]
    price_columns = ["European Call", "American Call", "European Put", "American Put"]

    for column in price_columns:
        convergence[f"{column} vs Ref"] = convergence[column] - reference[column]

    return convergence

# =========================================================
# 3.9 CRR Greeks Function
# =========================================================


def crr_tree_greeks(S, K, r, q, sigma, T, N, option_type, style):

    if N < 2:
        raise ValueError("At least 2 CRR steps are required for tree Greeks.")

    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    discount = math.exp(-r * dt)

    if not 0 <= p <= 1:
        raise ValueError(f"Invalid risk-neutral probability: {p}")

    j = np.arange(N + 1)
    stock_prices = S * u**j * d**(N - j)

    if option_type == "call":
        option_values = np.maximum(stock_prices - K, 0.0)
    elif option_type == "put":
        option_values = np.maximum(K - stock_prices, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    level_2 = None
    level_1 = None

    for i in range(N - 1, -1, -1):

        continuation = discount * ((1 - p) * option_values[:-1] + p * option_values[1:])

        if style == "american":

            j = np.arange(i + 1)
            stock_prices = S * u**j * d**(i - j)

            if option_type == "call":
                exercise = np.maximum(stock_prices - K, 0.0)
            else:
                exercise = np.maximum(K - stock_prices, 0.0)

            option_values = np.maximum(continuation, exercise)

        elif style == "european":
            option_values = continuation

        else:
            raise ValueError("style must be 'european' or 'american'")

        if i == 2:
            level_2 = option_values.copy()

        if i == 1:
            level_1 = option_values.copy()

    base_price = float(option_values[0])

    # -----------------------------------------------------
    # Delta
    # -----------------------------------------------------

    S_down = S * d
    S_up = S * u

    O_down = level_1[0]
    O_up = level_1[1]

    delta = (O_up - O_down) / (S_up - S_down)

    # -----------------------------------------------------
    # Gamma
    # -----------------------------------------------------

    S_dd = S * d**2
    S_du = S * d * u
    S_uu = S * u**2

    O_dd = level_2[0]
    O_du = level_2[1]
    O_uu = level_2[2]

    delta_down = (O_du - O_dd) / (S_du - S_dd)
    delta_up = (O_uu - O_du) / (S_uu - S_du)

    gamma = (delta_up - delta_down) / ((S_uu - S_dd) / 2)

    # -----------------------------------------------------
    # Theta
    # -----------------------------------------------------

    theta_annual = (O_du - base_price) / (2 * dt)
    theta_daily = theta_annual / 365

    return {
        "Price": base_price,
        "Delta": delta,
        "Gamma": gamma,
        "Theta Annual": theta_annual,
        "Theta Daily": theta_daily
    }

# =========================================================
# 3.10 Complete CRR Greeks Function
# =========================================================

def crr_greeks(S, K, r, q, sigma, T, N, option_type, style,
               vol_bump=0.005, rate_bump=0.001, dividend_bump=0.001):

    if sigma <= vol_bump:
        raise ValueError("Volatility is too low for the selected Vega bump.")

    tree_greeks = crr_tree_greeks(
        S=S, K=K, r=r, q=q, sigma=sigma, T=T, N=N,
        option_type=option_type, style=style
    )

    base_price = tree_greeks["Price"]
     # -----------------------------------------------------
    # Vega - price change per 1 volatility percentage point
    # -----------------------------------------------------

    price_vol_up = crr_price(
        S=S, K=K, r=r, q=q, sigma=sigma + vol_bump, T=T, N=N,
        option_type=option_type, style=style
    )

    price_vol_down = crr_price(
        S=S, K=K, r=r, q=q, sigma=sigma - vol_bump, T=T, N=N,
        option_type=option_type, style=style
    )

    vega_raw = (price_vol_up - price_vol_down) / (2 * vol_bump)
    vega = vega_raw * 0.01

      # -----------------------------------------------------
    # Rho - price change per 1 interest-rate percentage point
    # -----------------------------------------------------

    price_rate_up = crr_price(
        S=S, K=K, r=r + rate_bump, q=q, sigma=sigma, T=T, N=N,
        option_type=option_type, style=style
    )

    price_rate_down = crr_price(
        S=S, K=K, r=r - rate_bump, q=q, sigma=sigma, T=T, N=N,
        option_type=option_type, style=style
    )

    rho_raw = (price_rate_up - price_rate_down) / (2 * rate_bump)
    rho = rho_raw * 0.01

      # -----------------------------------------------------
    # Phi - price change per 1 dividend-yield percentage point
    # -----------------------------------------------------

    price_div_up = crr_price(
        S=S, K=K, r=r, q=q + dividend_bump, sigma=sigma, T=T, N=N,
        option_type=option_type, style=style
    )

    price_div_down = crr_price(
        S=S, K=K, r=r, q=q - dividend_bump, sigma=sigma, T=T, N=N,
        option_type=option_type, style=style
    )

    phi_raw = (price_div_up - price_div_down) / (2 * dividend_bump)
    phi = phi_raw * 0.01

       # -----------------------------------------------------
    # Omega / Elasticity
    # -----------------------------------------------------

    delta = tree_greeks["Delta"]
    omega = (delta * S / base_price) if base_price > 0 else None

    return {
        "Price": base_price,
        "Delta": delta,
        "Gamma": tree_greeks["Gamma"],
        "Theta Annual": tree_greeks["Theta Annual"],
        "Theta Daily": tree_greeks["Theta Daily"],
        "Vega": vega,
        "Rho": rho,
        "Phi": phi,
        "Omega": omega
    }


# =========================================================
# 4. MARKET DATA ACQUISITION
# =========================================================

# ---------------------------------------------------------
# 4.1 Ticker and spot price
# ---------------------------------------------------------

ticker = yf.Ticker(symbol)
history = ticker.history(period=recent_price_period, auto_adjust=False)

if history.empty or history["Close"].dropna().empty:
    raise ValueError(f"No recent price data returned for {symbol}.")

if spot_mode == "auto":
    spot = float(history["Close"].dropna().iloc[-1])
elif spot_mode == "manual":
    spot = float(manual_spot)
else:
    raise ValueError("spot_mode must be 'auto' or 'manual'.")

# ---------------------------------------------------------
# 4.2 Expiration selection
# ---------------------------------------------------------

expirations = ticker.options

if not expirations:
    raise ValueError(f"No option expirations returned for {symbol}.")

if show_available_expirations:
    print(f"Available expirations for {symbol}:")
    for i, exp in enumerate(expirations):
        print(f"{i:>2}: {exp}")

if expiration_mode == "index":
    if expiration_index < 0 or expiration_index >= len(expirations):
        raise IndexError(f"expiration_index {expiration_index} is outside the available range 0 to {len(expirations) - 1}.")
    expiration = expirations[expiration_index]
elif expiration_mode == "manual":
    expiration = manual_expiration
    if expiration not in expirations:
        raise ValueError(f"{expiration} is not an available expiration for {symbol}.")
else:
    raise ValueError("expiration_mode must be 'index' or 'manual'.")

# Likely need to print expiration chain to selected expiration and opiton strike

# ---------------------------------------------------------
# 4.3 Option chain and strike selection
# ---------------------------------------------------------

chain = ticker.option_chain(expiration)
calls = chain.calls
puts = chain.puts

near_spot_calls = calls[(calls["strike"] >= spot * (1 - near_spot_range)) & (calls["strike"] <= spot * (1 + near_spot_range))]

if show_near_spot_options:
    print(near_spot_calls[["strike", "bid", "ask", "lastPrice", "impliedVolatility"]])

call_match = calls[np.isclose(calls["strike"], K_market)]
put_match = puts[np.isclose(puts["strike"], K_market)]

if call_match.empty:
    raise ValueError(f"No call found at strike {K_market}.")
if put_match.empty:
    raise ValueError(f"No put found at strike {K_market}.")

selected_call = call_match.iloc[0]
selected_put = put_match.iloc[0]


# ---------------------------------------------------------
# 4.4 Option quote extraction and quality checks
# ---------------------------------------------------------

call_bid = float(selected_call["bid"])
call_ask = float(selected_call["ask"])
call_last = float(selected_call["lastPrice"])
call_iv = float(selected_call["impliedVolatility"])
put_bid = float(selected_put["bid"])
put_ask = float(selected_put["ask"])
put_last = float(selected_put["lastPrice"])
put_iv = float(selected_put["impliedVolatility"])

call_quote = quote_quality(call_bid, call_ask)
put_quote = quote_quality(put_bid, put_ask)
call_mid = call_quote["mid"]
put_mid = put_quote["mid"]


# ---------------------------------------------------------
# 4.5 Time to expiration
# ---------------------------------------------------------

today = pd.Timestamp.today().normalize()
expiration_date = pd.Timestamp(expiration)
days_to_expiration = (expiration_date - today).days

if days_to_expiration <= 0:
    raise ValueError("The selected option has expired or expires today.")

T_market = (days_to_expiration / 365)

# =========================================================
# MAnual CHecks
# =========================================================


#print(f"CRR Model Volatility: {sigma_model:.2%}")
#print(f"Latest Closing Price: ${spot:.2f}")
#print("Selected expiration:", expiration)
#print("Days to expiration:",days_to_expiration)


#print(f"Available expirations for {symbol}:")
#print()
#for i, expiration in enumerate(expirations):
    #print( f"{i:>2}: {expiration}")

#### ticker.live()

#print(near_spot_calls[ ["strike", "bid", "ask", "lastPrice"]])


# =========================================================
# 5. DYNAMIC MODEL ASSUMPTIONS
# =========================================================

# ---------------------------------------------------------
# 5.1 Dividend yield selection
# ---------------------------------------------------------


if dividend_mode == "auto":
    q_market, q_source = (get_dividend_yield(ticker,spot))
elif dividend_mode == "manual":
    q_market = manual_dividend_yield
    q_source = "Manual"
else:
    raise ValueError( "dividend_mode must be " "'auto' or 'manual'")

# ---------------------------------------------------------
# 5.2 Risk-free rate selection
# ---------------------------------------------------------


if rate_mode == "auto":
    treasury_rate = get_treasury_rate(days_to_expiration)
    r_market = treasury_rate["rate"]
    r_source = ("U.S. Treasury interpolated")

elif rate_mode == "manual":
    r_market = manual_risk_free_rate
    r_source = "Manual"

else:
    raise ValueError( "rate_mode must be 'auto' or 'manual'")

# ---------------------------------------------------------
# 5.3 Valuation volatility selection
# ---------------------------------------------------------

historical_vol = get_historical_volatility(ticker, historical_vol_period)

if volatility_mode == "historical":
    sigma_model = historical_vol
    sigma_source = ( "Historical Volatility")
elif volatility_mode == "manual":
    sigma_model = manual_volatility
    sigma_source = "Manual"
else:
    raise ValueError("volatility_mode must be 'historical' or 'manual'")
if sigma_model <= 0 or not np.isfinite(sigma_model):
    raise ValueError("Invalid model volatility.")

# =========================================================
# 6. CONSOLIDATED MARKET DATA OBJECT
# =========================================================

market_data = {
    "symbol": symbol,
    "spot": spot,
    "spot_source": "Yahoo recent close" if spot_mode == "auto" else "Manual",
    "expiration": expiration,
    "days_to_expiration": days_to_expiration,
    "T": T_market,
    "strike": K_market,
    "call_bid": call_bid,
    "call_ask": call_ask,
    "call_mid": call_mid,
    "call_last": call_last,
    "call_quote_status": call_quote["status"],
    "yahoo_call_iv": call_iv,
    "put_bid": put_bid,
    "put_ask": put_ask,
    "put_mid": put_mid,
    "put_last": put_last,
    "put_quote_status": put_quote["status"],
    "yahoo_put_iv": put_iv,
    "dividend_yield": q_market,
    "dividend_source": q_source,
    "risk_free_rate": r_market,
    "risk_free_source": r_source,
    "historical_volatility": historical_vol,
    "model_volatility": sigma_model,
    "model_volatility_source": sigma_source,
    "tree_steps": N_dynamic
}

# =========================================================
# 7. DYNAMIC CRR VALUATION
# =========================================================

dynamic_results = price_all_styles(S=spot, K=K_market, r=r_market, q=q_market, sigma=sigma_model, T=T_market, N=N_dynamic)

call_early_exercise = dynamic_results["American Call"] - dynamic_results["European Call"]
put_early_exercise = dynamic_results["American Put"] - dynamic_results["European Put"]
call_model_difference = dynamic_results["American Call"] - call_mid
put_model_difference = dynamic_results["American Put"] - put_mid

# =========================================================
# 8. CRR IMPLIED VOLATILITY ANALYSIS
# =========================================================

call_iv_market_price = call_mid
put_iv_market_price = put_mid

crr_call_iv = crr_implied_volatility(market_price=call_mid, S=spot, K=K_market, r=r_market, q=q_market, 
                                     T=T_market, N=N_dynamic, option_type="call", style=iv_style, 
                                     sigma_low=iv_sigma_low, sigma_high=iv_sigma_high)
crr_put_iv = crr_implied_volatility(market_price=put_mid, S=spot, K=K_market, r=r_market, q=q_market, 
                                    T=T_market, N=N_dynamic, option_type="put", style=iv_style, 
                                    sigma_low=iv_sigma_low, sigma_high=iv_sigma_high)

call_iv_check = None
put_iv_check = None

if crr_call_iv is not None:
    call_iv_check = crr_price(S=spot, K=K_market, r=r_market, q=q_market, sigma=crr_call_iv, T=T_market, 
                              N=N_dynamic, option_type="call", style=iv_style)

if crr_put_iv is not None:
    put_iv_check = crr_price(S=spot, K=K_market, r=r_market, q=q_market, sigma=crr_put_iv, T=T_market, 
                             N=N_dynamic, option_type="put", style=iv_style)


# --- Call IV range ---

crr_call_iv_bid = crr_implied_volatility(
    market_price=call_bid, S=spot, K=K_market,
    r=r_market, q=q_market, T=T_market, N=N_dynamic,
    option_type="call", style=iv_style,
    sigma_low=iv_sigma_low, sigma_high=iv_sigma_high
) if call_bid > 0 else None

crr_call_iv_ask = crr_implied_volatility(
    market_price=call_ask, S=spot, K=K_market,
    r=r_market, q=q_market, T=T_market, N=N_dynamic,
    option_type="call", style=iv_style,
    sigma_low=iv_sigma_low, sigma_high=iv_sigma_high
) if call_ask > 0 else None

# --- Put IV range ---

crr_put_iv_bid = crr_implied_volatility(
    market_price=put_bid, S=spot, K=K_market,
    r=r_market, q=q_market, T=T_market, N=N_dynamic,
    option_type="put", style=iv_style,
    sigma_low=iv_sigma_low, sigma_high=iv_sigma_high
) if put_bid > 0 else None

crr_put_iv_ask = crr_implied_volatility(
    market_price=put_ask, S=spot, K=K_market,
    r=r_market, q=q_market, T=T_market, N=N_dynamic,
    option_type="put", style=iv_style,
    sigma_low=iv_sigma_low, sigma_high=iv_sigma_high
) if put_ask > 0 else None

# =========================================================
# 9. CRR TREE CONVERGENCE AND NUMERICAL STABILITY
# =========================================================

convergence_table = crr_convergence_analysis(
    S=spot, K=K_market, r=r_market, q=q_market,
    sigma=sigma_model, T=T_market, steps=convergence_steps)

previous_tree = convergence_table.iloc[-2]
reference_tree = convergence_table.iloc[-1]

price_columns = ["European Call", "American Call", "European Put", "American Put"]

convergence_final_changes = {}

for column in price_columns:
    convergence_final_changes[column] = abs(reference_tree[column] - previous_tree[column])

convergence_pass = all(change <= convergence_tolerance for change in convergence_final_changes.values())

# =========================================================
# 10. CRR GREEKS ANALYSIS
# =========================================================

if greek_volatility_mode == "model":

    greek_call_sigma = sigma_model
    greek_put_sigma = sigma_model
    greek_sigma_source = "Model Volatility"

elif greek_volatility_mode == "crr_iv":

    if crr_call_iv is None or crr_put_iv is None:
        raise ValueError("CRR implied volatility is unavailable for Greeks.")

    greek_call_sigma = crr_call_iv
    greek_put_sigma = crr_put_iv
    greek_sigma_source = "CRR Implied Volatility"

else:
    raise ValueError("greek_volatility_mode must be 'model' or 'crr_iv'")

european_call_greeks = crr_greeks(
    S=spot, K=K_market, r=r_market, q=q_market, sigma=greek_call_sigma,
    T=T_market, N=greek_steps, option_type="call", style="european",
    vol_bump=greek_vol_bump, rate_bump=greek_rate_bump, dividend_bump=greek_dividend_bump
)

american_call_greeks = crr_greeks(
    S=spot, K=K_market, r=r_market, q=q_market, sigma=greek_call_sigma,
    T=T_market, N=greek_steps, option_type="call", style="american",
    vol_bump=greek_vol_bump, rate_bump=greek_rate_bump, dividend_bump=greek_dividend_bump
)

european_put_greeks = crr_greeks(
    S=spot, K=K_market, r=r_market, q=q_market, sigma=greek_put_sigma,
    T=T_market, N=greek_steps, option_type="put", style="european",
    vol_bump=greek_vol_bump, rate_bump=greek_rate_bump, dividend_bump=greek_dividend_bump
)

american_put_greeks = crr_greeks(
    S=spot, K=K_market, r=r_market, q=q_market, sigma=greek_put_sigma,
    T=T_market, N=greek_steps, option_type="put", style="american",
    vol_bump=greek_vol_bump, rate_bump=greek_rate_bump, dividend_bump=greek_dividend_bump
)

greeks_table = pd.DataFrame({
    "European Call": european_call_greeks,
    "American Call": american_call_greeks,
    "European Put": european_put_greeks,
    "American Put": american_put_greeks
}).T

# =========================================================
# 11. DASHBOARD DATA PACKAGE
# =========================================================

dashboard_data = {
    "symbol": symbol,
    "spot": spot,
    "strike": K_market,
    "expiration": expiration,
    "days_to_expiration": days_to_expiration,
    "T": T_market,
    "tree_steps": N_dynamic,

    "risk_free_rate": r_market,
    "risk_free_source": r_source,
    "dividend_yield": q_market,
    "dividend_source": q_source,

    "historical_volatility": historical_vol,
    "model_volatility": sigma_model,
    "model_volatility_source": sigma_source,

    "call_bid": call_bid,
    "call_mid": call_mid,
    "call_ask": call_ask,
    "put_bid": put_bid,
    "put_mid": put_mid,
    "put_ask": put_ask,

    "yahoo_call_iv": call_iv,
    "yahoo_put_iv": put_iv,
    "crr_call_iv": crr_call_iv,
    "crr_put_iv": crr_put_iv,

    "crr_results": dynamic_results,
    "convergence": convergence_table,
    "greeks": greeks_table
}

# =========================================================
# GREEK VALIDATION
# =========================================================

greek_checks = [
    ("European Call Delta", 0 <= european_call_greeks["Delta"] <= 1),
    ("American Call Delta", 0 <= american_call_greeks["Delta"] <= 1),
    ("European Put Delta", -1 <= european_put_greeks["Delta"] <= 0),
    ("American Put Delta", -1 <= american_put_greeks["Delta"] <= 0),
    ("European Call Gamma", european_call_greeks["Gamma"] >= 0),
    ("American Call Gamma", american_call_greeks["Gamma"] >= 0),
    ("European Put Gamma", european_put_greeks["Gamma"] >= 0),
    ("American Put Gamma", american_put_greeks["Gamma"] >= 0),
    ("European Call Vega", european_call_greeks["Vega"] >= 0),
    ("European Put Vega", european_put_greeks["Vega"] >= 0)
]


# =========================================================
# 12. MODEL OUTPUT AND DIAGNOSTICS
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("ADVANCED DYNAMIC CRR BINOMIAL MODEL")
    print("=" * 60)
    print(f"Ticker                  : {symbol}")
    print(f"Spot                    : ${spot:,.2f}")
    print(f"Strike                  : ${K_market:,.2f}")
    print(f"Expiration              : {expiration}")
    print(f"Days to Expiration      : {days_to_expiration}")
    print(f"Tree Steps              : {N_dynamic:,}")
    print()
    print(f"Dividend Yield          : {q_market:.4%} | {q_source}")
    print(f"Risk-Free Rate          : {r_market:.4%} | {r_source}")
    print(f"Historical Volatility   : {historical_vol:.2%}")
    print(f"Model Volatility        : {sigma_model:.2%} | {sigma_source}")
    print()
    print("-" * 60)
    print("OPTION MARKET DATA")
    print("-" * 60)
    print(f"Call Bid / Ask / Mid    : ${call_bid:,.2f} / ${call_ask:,.2f} / ${call_mid:,.2f} | {call_quote['status']}")
    print(f"Put Bid / Ask / Mid     : ${put_bid:,.2f} / ${put_ask:,.2f} / ${put_mid:,.2f} | {put_quote['status']}")
    print(f"Yahoo Call IV           : {call_iv:.2%}")
    print(f"Yahoo Put IV            : {put_iv:.2%}")
    print()
    print("-" * 60)
    print("CRR MODEL RESULTS")
    print("-" * 60)

    for name, price in dynamic_results.items():
        print(f"{name:<24}: ${price:,.4f}")


    print()
    print(f"Call Early Exercise     : ${call_early_exercise:,.4f}")
    print(f"Put Early Exercise      : ${put_early_exercise:,.4f}")
    print(f"Call Model - Market     : ${call_model_difference:,.4f}")
    print(f"Put Model - Market      : ${put_model_difference:,.4f}")
    print()

    ### Volatility Analysis
    print()
    print("=" * 60)
    print("VOLATILITY ANALYSIS")
    print("=" * 60)

    print(f"Historical Volatility   : {historical_vol:.4%}")
    print(f"Model Volatility        : {sigma_model:.4%} | {sigma_source}")
    print()
    print(f"Yahoo Call IV           : {call_iv:.4%}")
    print(f"Yahoo Put IV            : {put_iv:.4%}")

    print()
    print("=" * 60)
    print("CRR IMPLIED VOLATILITY")
    print("=" * 60)

    print(f"IV Pricing Style        : {iv_style.title()}")
    print(f"Tree Steps              : {N_dynamic:,}")

    print()
    print("CALL")
    print("-" * 60)
    print(f"Market Bid              : ${call_bid:,.4f}")
    print(f"Market Mid              : ${call_mid:,.4f}")
    print(f"Market Ask              : ${call_ask:,.4f}")
    print(f"Yahoo IV                : {call_iv:.4%}")
    print(f"CRR IV - Bid            : {crr_call_iv_bid:.4%}" if crr_call_iv_bid is not None else "CRR IV - Bid            : N/A")
    print(f"CRR IV - Mid            : {crr_call_iv:.4%}" if crr_call_iv is not None else "CRR IV - Mid            : N/A")
    print(f"CRR IV - Ask            : {crr_call_iv_ask:.4%}" if crr_call_iv_ask is not None else "CRR IV - Ask            : N/A")
    print(f"CRR Price @ Mid IV      : ${call_iv_check:,.4f}" if call_iv_check is not None else "CRR Price @ Mid IV      : N/A")

    print()
    print("PUT")
    print("-" * 60)
    print(f"Market Bid              : ${put_bid:,.4f}")
    print(f"Market Mid              : ${put_mid:,.4f}")
    print(f"Market Ask              : ${put_ask:,.4f}")
    print(f"Yahoo IV                : {put_iv:.4%}")
    print(f"CRR IV - Bid            : {crr_put_iv_bid:.4%}" if crr_put_iv_bid is not None else "CRR IV - Bid            : N/A")
    print(f"CRR IV - Mid            : {crr_put_iv:.4%}" if crr_put_iv is not None else "CRR IV - Mid            : N/A")
    print(f"CRR IV - Ask            : {crr_put_iv_ask:.4%}" if crr_put_iv_ask is not None else "CRR IV - Ask            : N/A")
    print(f"CRR Price @ Mid IV      : ${put_iv_check:,.4f}" if put_iv_check is not None else "CRR Price @ Mid IV      : N/A")

    if treasury_rate is not None:
        print()
        print(f"Treasury Observation    : {treasury_rate['date'].date()}")

    print()
    print("=" * 78)
    print("CRR TREE CONVERGENCE")
    print("=" * 78)

    display_columns = ["Steps", "European Call", "American Call", "European Put", "American Put"]

    print(
        convergence_table[display_columns].to_string(
            index=False,
            formatters={
                "European Call": lambda x: f"${x:,.4f}",
                "American Call": lambda x: f"${x:,.4f}",
                "European Put": lambda x: f"${x:,.4f}",
                "American Put": lambda x: f"${x:,.4f}"
            }
        )
    )

    print()
    print("-" * 78)
    print(f"REFERENCE TREE: {int(reference_tree['Steps']):,} STEPS")
    print("-" * 78)

    for column in price_columns:
        print(f"{column:<18}: ${reference_tree[column]:,.4f}")

    print()
    print(f"CHANGE FROM {int(previous_tree['Steps']):,} TO {int(reference_tree['Steps']):,} STEPS")
    print("-" * 78)

    for column, change in convergence_final_changes.items():
        print(f"{column:<18}: ${change:,.6f}")

    print()
    print(f"Tolerance            : ${convergence_tolerance:.4f}")
    print(f"Convergence Status   : {'PASS' if convergence_pass else 'REVIEW'}")

    print()
    print("-" * 78)
    print("DISTANCE FROM REFERENCE TREE")
    print("-" * 78)

    reference_columns = [
        "Steps",
        "European Call vs Ref",
        "American Call vs Ref",
        "European Put vs Ref",
        "American Put vs Ref"
    ]

    print(
        convergence_table[reference_columns].to_string(
            index=False,
            formatters={
                "European Call vs Ref": lambda x: f"{x:+.6f}",
                "American Call vs Ref": lambda x: f"{x:+.6f}",
                "European Put vs Ref": lambda x: f"{x:+.6f}",
                "American Put vs Ref": lambda x: f"{x:+.6f}"
            }
        )
    )

    print()
    print("=" * 105)
    print("CRR GREEKS")
    print("=" * 105)

    print(f"Volatility Source : {greek_sigma_source}")
    print(f"Call Volatility   : {greek_call_sigma:.4%}")
    print(f"Put Volatility    : {greek_put_sigma:.4%}")
    print(f"Tree Steps        : {greek_steps:,}")

    print()
    print(
        greeks_table.to_string(
            formatters={
                "Price": lambda x: f"${x:,.4f}",
                "Delta": lambda x: f"{x:.6f}",
                "Gamma": lambda x: f"{x:.6f}",
                "Theta Annual": lambda x: f"${x:,.4f}",
                "Theta Daily": lambda x: f"${x:,.6f}",
                "Vega": lambda x: f"${x:,.6f}",
                "Rho": lambda x: f"${x:,.6f}",
                "Phi": lambda x: f"${x:,.6f}",
                "Omega": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            }
        )
    )

    print()
    print("-" * 60)
    print("GREEK SANITY CHECKS")
    print("-" * 60)

    for check_name, passed in greek_checks:
        print(f"{check_name:<28}: {'PASS' if passed else 'REVIEW'}")




