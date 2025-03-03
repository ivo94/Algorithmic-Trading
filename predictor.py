import ccxt
import pandas as pd
from prophet import Prophet
from scipy.stats import norm
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
import itertools
import numpy as np

class PredictionResult(dict):
    def __str__(self):
        header = f"\n{' Prediction Results ':=^80}\n"
        symbol_info = f"Symbol: {self.get('symbol', 'N/A')} ({self.get('timeframe', 'N/A')})\n"
        current_price = f"Current Price: {self.get('current_price', 'N/A'):.4f}\n"
        
        # Performance metrics
        coverage = f"\n[Model Coverage]\nInterval Coverage: {self.get('coverage', 0):.2f}%\n"
        
        # Accuracy section
        acc = self.get('total_accuracy', {})
        inc = self.get('increase_accuracy', {})
        dec = self.get('decrease_accuracy', {})
        
        accuracy = (
            f"\n[Prediction Accuracy] (Threshold: {self.get('threshold', 0)*100:.0f}% of predicted move)\n"
            f"Total Accuracy: {acc.get('percentage', 0):.2f}% ({acc.get('correct', 0)}/{acc.get('total', 0)})\n"
            f"  - Increases: {inc.get('percentage', 0):.2f}% ({inc.get('correct', 0)}/{inc.get('total', 0)})\n"
            f"  - Decreases: {dec.get('percentage', 0):.2f}% ({dec.get('correct', 0)}/{dec.get('total', 0)})\n"
        )
        
        # CV settings
        cv = self.get('cv_settings', {})
        cv_settings = (
            f"\n[Cross-Validation Settings]\n"
            f"Training Period: {cv.get('initial', 'N/A')}\n"
            f"Validation Window: {cv.get('period', 'N/A')}\n"
            f"Prediction Horizon: {cv.get('horizon', 'N/A')}\n"
        )
        
        # Forecast
        forecast = "\n[Latest Forecast]\n"
        if not self.get('forecast', None).empty:
            forecast += self['forecast'].to_string(index=False, float_format="%.4f")
        else:
            forecast += "No forecast available"
        
        return (
            header + 
            symbol_info + 
            current_price + 
            coverage + 
            accuracy + 
            cv_settings + 
            forecast + 
            "\n" + "="*80 + "\n"
        )


def parse_timeframe(timeframe):
    import re
    match = re.match(r'^(\d*)([mhdw])$', timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    value_part = match.group(1)
    unit = match.group(2)
    if not value_part:
        value = 1
    else:
        value = int(value_part)
    return value, unit

def get_pandas_freq(timeframe):
    value, unit = parse_timeframe(timeframe)
    unit_map = {
        'm': 'T',
        'h': 'H',
        'd': 'D',
        'w': 'W'
    }
    pandas_unit = unit_map[unit]
    return f"{value}{pandas_unit}"

def predict(symbol, timeframe='1d', threshold=0.5, period='1h', horizon='1h',
            changepoint_prior_scale=1, 
            daily_seasonality=False,
            seasonality_prior_scale=10.0,
            seasonality_mode='additive',
            n_changepoints=25,
            interval_width=0.99,
            initial_period_candles=500):
    # Ensure threshold is numeric
    try:
        threshold = float(threshold)
    except (ValueError, TypeError):
        raise TypeError("Threshold parameter must be a numeric value")

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'adjustForTimeDifference': True},
    })

    limit = 3000
    initial_period_candles = 700

    # Fetch current price
    ticker = exchange.fetch_ticker(symbol)
    current_price = ticker['last']

    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Prepare Prophet data
    prophet_df = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
    pandas_freq = get_pandas_freq(timeframe)
    prophet_df = prophet_df.set_index('ds').asfreq(pandas_freq).reset_index()
    prophet_df['y'] = prophet_df['y'].interpolate(method='linear').astype(float)

    # Fit model with parameterized settings
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        daily_seasonality=daily_seasonality,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode,
        n_changepoints=n_changepoints,
        interval_width=interval_width,
        mcmc_samples=0,
        uncertainty_samples=1000
    )

    model.fit(prophet_df)

    # Parse timeframe components
    timeframe_value, timeframe_unit = parse_timeframe(timeframe)
    base_unit_plural = {'m': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}[timeframe_unit]

    # Configure cross-validation parameters
    def parse_cv_param(param, default_value, default_unit):
        if param is None:
            return f"{default_value} {default_unit}"
        val, unit = parse_timeframe(param)
        unit_plural = {'m': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}[unit]
        return f"{val} {unit_plural}"

    period_str = parse_cv_param(period, timeframe_value, base_unit_plural)
    horizon_str = parse_cv_param(horizon, timeframe_value, base_unit_plural)

    # Cross-validation with retry logic
    success = False
    while initial_period_candles > 0:
        try:
            duration = initial_period_candles * timeframe_value
            initial_str = f"{duration} {base_unit_plural}"
            
            df_cv = cross_validation(
                model,
                initial=initial_str,
                period=period_str,
                horizon=horizon_str,
                # parallel='processes'
            )
            success = True
            break
        except ValueError as e:
            print(f"Failed with initial = {duration} {base_unit_plural}. Error: {e}")
            initial_period_candles -= 1

    if not success:
        print("All retries failed. Could not complete cross-validation.")
        return None

    # Calculate coverage percentage
    df_cv['in_interval'] = (df_cv['y'].astype(float) >= df_cv['yhat_lower'].astype(float)) & \
                          (df_cv['y'].astype(float) <= df_cv['yhat_upper'].astype(float))
    coverage_percentage = df_cv['in_interval'].mean() * 100

    # Merge with previous prices
    df_cv = df_cv.merge(
        prophet_df[['ds', 'y']], 
        left_on='cutoff',
        right_on='ds',
        suffixes=('', '_prev')
    ).rename(columns={'y_prev': 'y_prev'}).drop(columns=['ds_prev'])

    # Convert to numeric types
    df_cv['y_prev'] = pd.to_numeric(df_cv['y_prev'], errors='coerce')
    df_cv['yhat'] = pd.to_numeric(df_cv['yhat'], errors='coerce')
    df_cv['y'] = pd.to_numeric(df_cv['y'], errors='coerce')

    # Calculate price changes
    df_cv['predicted_change'] = df_cv['yhat'] - df_cv['y_prev']
    df_cv['actual_change'] = df_cv['y'] - df_cv['y_prev']

    # Calculate directional accuracy
    increase_mask = df_cv['predicted_change'] > 0
    decrease_mask = df_cv['predicted_change'] < 0
    
    # Ensure numeric types
    predicted_changes_increase = df_cv[increase_mask]['predicted_change'].astype(float)
    actual_changes_increase = df_cv[increase_mask]['actual_change'].astype(float)
    correct_increases = (actual_changes_increase >= threshold * predicted_changes_increase).sum()
    total_increases = increase_mask.sum()
    
    predicted_changes_decrease = df_cv[decrease_mask]['predicted_change'].astype(float)
    actual_changes_decrease = df_cv[decrease_mask]['actual_change'].astype(float)
    correct_decreases = (actual_changes_decrease <= threshold * predicted_changes_decrease).sum()
    total_decreases = decrease_mask.sum()

    # Calculate total accuracy
    total_correct = correct_increases + correct_decreases
    total_predictions = total_increases + total_decreases
    total_percentage = (total_correct / total_predictions * 100) if total_predictions > 0 else 0

    # Generate future forecast
    future = model.make_future_dataframe(periods=1, freq=pandas_freq)
    forecast = model.predict(future)
    future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]

    return PredictionResult({
        'symbol': symbol,
        'timeframe': timeframe,
        'threshold': threshold,
        'coverage': coverage_percentage,
        'forecast': future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        'current_price': current_price,
        'increase_accuracy': {
            'correct': int(correct_increases),
            'total': int(total_increases),
            'percentage': correct_increases/total_increases*100 if total_increases > 0 else 0
        },
        'decrease_accuracy': {
            'correct': int(correct_decreases),
            'total': int(total_decreases),
            'percentage': correct_decreases/total_decreases*100 if total_decreases > 0 else 0
        },
        'total_accuracy': {
            'correct': int(total_correct),
            'total': int(total_predictions),
            'percentage': total_percentage
        },
        'cv_settings': {
            'initial': f"{initial_period_candles * timeframe_value} {base_unit_plural}",
            'period': period_str,
            'horizon': horizon_str
        }
    })

def tune_prophet(symbol, timeframe='1d', threshold=0.5, 
                param_grid=None, n_samples=10):
    # Define default parameter grid
    if param_grid is None:
        param_grid = {
            'changepoint_prior_scale': np.logspace(-2, 0, 5),  # 0.01 to 1
            'daily_seasonality': [True, False],
            'seasonality_prior_scale': [5, 10, 20],
            'seasonality_mode': ['additive', 'multiplicative']
        }
    
    best_score = -np.inf
    best_params = {}
    
    # Generate parameter combinations
    keys, values = zip(*param_grid.items())
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        
        try:
            result = predict(
                symbol=symbol,
                timeframe=timeframe,
                threshold=threshold,
                **params
            )
            
            if result and result['total_accuracy']['percentage'] > best_score:
                best_score = result['total_accuracy']['percentage']
                best_params = params
                
        except Exception as e:
            print(f"Failed with {params}: {str(e)}")
            continue
            
    print(f"\n🏆 Best Accuracy: {best_score:.2f}%")
    print("⚙️ Optimal Parameters:")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")
    
    return best_params, best_score


# Initialize the Binance exchange instance
def predict2(symbol):
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'adjustForTimeDifference': True},
    })

    timeframe = '1d'
    limit = 3000
    initial_period = 500
    # Fetch the ticker information
    ticker = exchange.fetch_ticker(symbol)
    # Extract the current price
    current_price = ticker['last']

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Prepare data for Prophet
    prophet_df = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
    prophet_df = prophet_df.set_index('ds').asfreq('D').reset_index()
    prophet_df['y'] = prophet_df['y'].interpolate(method='linear')

    # Step 2: Fit the Model
    model = Prophet(changepoint_prior_scale=0.5, daily_seasonality=False, interval_width=0.95, mcmc_samples=0, uncertainty_samples=1000)
    model.fit(prophet_df)

    # Step 3: Make Future Predictions
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)

    # Step 4: Perform Cross-Validation
    while initial_period > 0:
        try:
            df_cv = cross_validation(
                model,
                initial=f"{initial_period} days",
                period='5 days',
                horizon='1 days'
            )
            print("Cross-validation succeeded!")
            break  # Exit the loop if successful
        except ValueError as e:
            print(f"Failed with initial = {initial_period} days. Error: {e}")
            initial_period -= 1  # Reduce the initial value and retry

    # If exhausted all retries
    if initial_period <= 0:
        print("All retries failed. Could not complete cross-validation.")


    # Step 5: Compute Performance Metrics
    df_p = performance_metrics(df_cv)

    # Step 2: Calculate coverage percentage
    df_cv['in_interval'] = ((df_cv['y'] >= df_cv['yhat_lower']) & (df_cv['y'] <= df_cv['yhat_upper']))

    # Overall coverage
    coverage_percentage = df_cv['in_interval'].mean() * 100
    print(f'Overall Coverage Percentage for {symbol}: {coverage_percentage:.2f}%')

    # Coverage per day in horizon
    df_cv['day'] = (df_cv['ds'] - df_cv['cutoff']).dt.days
    coverage_by_day = df_cv.groupby('day')['in_interval'].mean() * 100
    print(coverage_by_day)

    # Extract the predictions for the future periods only
    future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]
    future_forecast = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Print the predicted prices with intervals
    print(f"\nPredicted Prices and Intervals for {symbol}:")
    print(future_forecast.to_string(index=False))

    # Step 2: Incorporate coverage into existing plot
    # Plot actual vs predicted prices
    # plt.figure(figsize=(12, 6))
    # plt.plot(prophet_df['ds'], prophet_df['y'], label='Actual Price', color='blue', linewidth=2)
    # plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Price', color='orange', linewidth=2, linestyle='--')
    # plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
    #                 color='gray', alpha=0.3, label='Prediction Interval')
    # plt.title('Actual vs Predicted Prices with Coverage')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.legend()

    # # Add text box with overall coverage percentage
    # textstr = f'Overall Coverage: {coverage_percentage:.2f}%'
    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
    #         verticalalignment='top', bbox=props)

    # plt.grid(True)
    # plt.show()

    return [coverage_percentage,future_forecast,current_price]

# Lower and higher estimates, average and current price
# Calculates the expected return based on the probability that the current price reaches a greater
# value than average and the expected profit if it did
def expected_return(low,high,avg,curr,coverage_percentage):
    profit = avg-curr
    mu = avg
    sigma = (high-low) / 2*1.96
    norm_prob = 1-norm.cdf(avg,mu,sigma)
    probability = norm_prob * coverage_percentage/100
    exp_return = profit * probability*100/curr
    return exp_return

def calcualte_invest(expected_returns,initial_amount):
    total = sum(expected_returns)
    fractions = [expected_returns[i]*100/total for i in range(len(expected_returns))]
    # fractions.sort(reverse=True)
    investments = [fractions[i]*initial_amount for i in range(len(fractions))]
    investments.sort(reverse=True)
    return investments

def single_model():
    symbol = input('Symbol: ')
    timeframe = input('Timeframe: ')
    threshold = input('Threshold: ')
    print(predict(symbol,timeframe,threshold=threshold))
    # tune_prophet(symbol=symbol,timeframe=timeframe,threshold=threshold)

# def single_model2():
#     symbol = input('Symbol: ')
#     predict2(symbol)

def main():
    initial_amount = int(input("Initial amount: "))
    tokens = ['BTC/USDT','ETH/USDT','XRP/USDT','SOL/USDT','BNB/USDT','LTC/USDT','DOGE/USDT','CAKE/USDT','ADA/USDT','PEPE/USDT']
    tokens2 = ['BTC/USDT','ETH/USDT']
    coverages = []
    expected_returns = []
    for token in tokens2:
        res = predict(token)
        coverage_percentage = res[0]
        future_forecast = res[1]
        current_price = res[2]
        low = future_forecast['yhat_lower']
        high = future_forecast['yhat_upper']
        avg = future_forecast['yhat']
        ret = expected_return(low,high,avg,current_price,coverage_percentage)
        print(f"expected_return = {ret}")
        expected_returns.append((ret,token))
        coverages.append((coverage_percentage,token))
        coverages.sort(key=lambda x: x[0],reverse=True)
        expected_returns.sort(key=lambda x: x[0],reverse=True)

    print(coverages)
    print(expected_returns)
    print(calcualte_invest(expected_returns,initial_amount))

# main()
single_model()