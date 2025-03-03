import ccxt
import time
import sys
import os

API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')

# counter = 0
# profit = 0

# Example: 'BTC/USDT', returns BTC
def get_base_currency(s):
    return s.split('/')[0]

# Amount to invest represents how much money the user is willing to use for the cycle of market making
# i.e the user wants to use 100 USDT as the initial amount
# It will use all of it to buy as much USDC as possible
def market_making(amount_to_invest,symbol):
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',  # Ensure we're trading on the spot market
        },
    })
    # Counts the cycles that it managed to execute
    # A cycle is a successful buy order followed by a successful sell order
    counter = 0
    profit = 0
    # Load market
    markets = exchange.load_markets()
    # This flag indicates if it is time to buy or to sell
    to_buy = True
    # Ensure limit orders don't execute immediately as taker orders to avoid fees
    params = {'postOnly': True}
    # Price precision, minimum unit
    tick_size = exchange.market(symbol)['precision']['price']
    # Start cycle
    while True:
        # Fetch symbol book
        order_book = exchange.fetch_order_book(symbol)
        if to_buy:
            # Get best bid
            current_best_bid = order_book['bids'][0][0]
            buy_price = current_best_bid
            buy_amount = amount_to_invest/buy_price
            buy_amount = float(exchange.amount_to_precision(symbol,buy_amount))
            try:
                buy_order = exchange.create_limit_buy_order(symbol,buy_amount,buy_price,params)
                # buy_order = exchange.create_limit_buy_order(symbol,buy_amount,buy_price,{'postOnly': True, 'FOK': True})
                
            except ccxt.ExchangeError as e:
                error_message = str(e)
                print(f"An error occurred while creating the buy order: {e}")
                if 'immediately match and take' in error_message.lower():
                        # Immediately fetch new best bid and try again
                        continue
                elif 'binance account has insufficient balance for requested action' in error_message.lower():
                    print(f"Buy amount = {buy_amount}")
                    print(f"Buy price = {buy_price}")
                    sys.exit()
            while to_buy:
                order_book = exchange.fetch_order_book(symbol)
                current_best_bid = order_book['bids'][0][0]
                buy_order_details = exchange.fetch_order(buy_order['id'],symbol)
                buy_order_status = buy_order_details['status']
                if (current_best_bid != buy_price): 
                    if (buy_order_status == 'open' and float(buy_order_details['filled'])) == 0:
                        try:
                            exchange.cancel_order(buy_order['id'], symbol)
                            break
                        except ccxt.OrderNotFound:
                            to_buy = False
                            break
                    elif buy_order_status == 'open' and buy_order_details['filled'] > 0:
                        while True:
                            buy_order_details = exchange.fetch_order(buy_order['id'],symbol)
                            buy_order_status = buy_order_details['status']
                            if buy_order_status == 'open':
                                print("Order has not been filled completly")
                                continue
                            else:
                                to_buy = False
                                break
                    elif buy_order_status == 'closed':
                        to_buy = False
                    else:
                        print("Order was rejected, expired, partially filled, or was canceled")
        else:
            current_best_ask = order_book['asks'][0][0]
            # some_ask = order_book['asks'][3][0]
            # Represents the minimum price needed to make a profit based on the buy price
            # FALTAn LAS COMISIONES
            # POR AHORA UTILIZAR SOLO SIN COMISIONES
            # Para obtener las fees: fees = exchange.fetch_trading_fees()
            # Output format: 
            '''
            {
            'maker': 0.001,  # 0.1% fee for maker orders (limit orders)
            'taker': 0.001   # 0.1% fee for taker orders (market orders)
            }
            '''
            minimum_profit_price = max(buy_price + tick_size, current_best_ask)
            sell_price = float(exchange.price_to_precision(symbol, minimum_profit_price))
            sell_amount = buy_amount
            try:
                sell_order = exchange.create_limit_sell_order(symbol,sell_amount,sell_price,params)
            except ccxt.ExchangeError as e:
                error_message = str(e)
                print(f"An error occurred while creating the sell order: {e}")
                if 'immediately match and take' in error_message.lower():
                        continue
                elif 'binance account has insufficient balance for requested action' in error_message.lower():
                    print(f"Sell amount = {sell_amount}")
                    print(f"Sell price = {sell_price}")
                    sys.exit()
            while not to_buy:
                sell_order_details = exchange.fetch_order(sell_order['id'],symbol)
                sell_order_status = sell_order_details['status']
                if sell_order_status == 'open':
                    continue
                elif sell_order_status == 'closed':
                    to_buy = True
                    counter += 1
                    absolute_profit = float(sell_price) * float(sell_amount) - float(buy_price) * float(buy_amount)
                    # absolute_profit = exchange.amount_to_precision(symbol,absolute_profit)
                    profit += absolute_profit
                    base_currency = get_base_currency(symbol)
                    print("Successful cycle completed")
                    print(f"Absolute cycle profit = {absolute_profit} USDT ")
                    print(f"Percentage cycle profit = {absolute_profit * 100/(float(buy_amount) * float(buy_price))} %")
                    print(f"{counter} cycles have been completed so far")
                    print(f"Total profit = {profit} USDT")
                    print(f"Total percentage profit = {profit * 100/amount_to_invest} %")
                else:
                    print("Order was rejected, expired, or was canceled")

def main():
    symbol = input("Symbol (e.g., BTC/USDT): ").upper()
    initial_amount = float(input("Initial Amount: "))
    start_time = time.time()
    try:
        market_making(initial_amount,symbol)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nScript stopped. Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        # print(f"{counter} cycles have been completed so far")
        # print(f"Total profit = {profit} USDT")
        # print(f"Total profit = {profit} USDT")

if __name__ == "__main__":
    main()