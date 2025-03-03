import ccxt
import time
import pytz
import datetime
import os

API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')

symbol = 'RED/USDT'  # The new trading pair

exchange = ccxt.binance({
    'apiKey': API_KEY,  # Replace with your API key
    'secret': API_SECRET,  # Replace with your secret
    'enableRateLimit': True,
    'options': {
        'adjustForTimeDifference': True,
        'defaultType': 'spot',
    },
})

exchange.load_markets(reload=True)

# Define your buy orders with different prices and total USDT to spend
buy_orders = [
    {'price': 0.1083, 'total_quote_amount': 152},
    {'price': 0.0685, 'total_quote_amount': 132},
    {'price': 0.0571, 'total_quote_amount': 50},
    {'price': 0.0476, 'total_quote_amount': 26},
    {'price': 0.0397, 'total_quote_amount': 20},
]

# Example buy orders for testing
# buy_orders = [
#     {'price': 1.0008, 'total_quote_amount': 10},
#     {'price': 1.0007, 'total_quote_amount': 10},
#     {'price': 1.0006, 'total_quote_amount': 10},
#     {'price': 1.0005, 'total_quote_amount': 10},
#     {'price': 1.000400, 'total_quote_amount': 10},
# ]

# Pre-compute adjusted prices and amounts for buy orders
def prepare_buy_orders(buy_orders):
    prepared_orders = []
    for order in buy_orders:
        price = float(order['price'])
        total_quote_amount = float(order['total_quote_amount'])
        # Calculate amount
        amount = total_quote_amount / price
        # Append to prepared orders
        prepared_orders.append({'price': price, 'amount': amount})

    return prepared_orders

prepared_buy_orders = prepare_buy_orders(buy_orders)

# Time setup
argentina_tz = pytz.timezone('America/Argentina/Buenos_Aires')
# Set the exact listing time
listing_time = datetime.datetime.now(argentina_tz).replace(hour=3, minute=14, second=0, microsecond=0)

# Wait until 7:00 am Argentina time
now_argentina = datetime.datetime.now(argentina_tz)
time_diff = (listing_time - now_argentina).total_seconds()

if time_diff > 0:
    print(f"Waiting for {time_diff} seconds until listing Argentina time.")
    time.sleep(time_diff)

# At this point, it's 7:00 am Argentina time
# print("It's listing time. Attempting to place orders.")

# Function to attempt placing orders until successful
def attempt_place_orders():
    # Load markets until symbol is available
    while True:
        # print("Inside the loop")
        try:
            # Ultra-lightweight check (order book endpoint)
            exchange.public_get_depth({'symbol': symbol.replace('/', '')})
            # print(f"{symbol} detected! Proceeding...")
            break
        except ccxt.BadSymbol:
            print("Not listed yet")
            time.sleep(1)
            pass  # Not listed yet
        except Exception as e:
            print(f"Error: {str(e)[:100]}")  # Truncate long messages

    # Place prepared limit buy orders
    for order in prepared_buy_orders:
        price = order['price']
        amount = order['amount']
        try:
            # Adjust price and amount to exchange precision
            price_adj = float(exchange.price_to_precision(symbol, price))
            amount_adj = float(exchange.amount_to_precision(symbol, amount))

            # Place limit buy order
            buy_order = exchange.create_order(
                symbol=symbol,
                type='limit',
                side='buy',
                amount=amount,
                price=price,
                params={'postOnly': True}
            )
            # print(f"Limit buy order placed successfully at price {price_adj} for amount {amount_adj}:")
            # print(buy_order)
        except Exception as e:
            print(f"Error placing limit buy order at price {price}:")
            print(e)


# def sell_all_red_balance(target_price=None):
#     balance = exchange.fetch_balance().get('RED', {}).get('free', 0)
#     while balance > 0:
#         try:
#             # 1. Get current RED balance
#             if balance <= 1e-6:  # Check for dust amounts
#                 print("No RED balance to sell")
#                 return True

#             # 3. Prepare order parameters
#             sell_price = float(exchange.price_to_precision(symbol, target_price))
#             sell_amount = float(exchange.amount_to_precision(symbol, balance))

#             # 4. Place limit sell order
#             order = exchange.create_order(
#                 symbol=symbol,
#                 type='limit',
#                 side='sell',
#                 amount=sell_amount,
#                 price=sell_price,
#                 params={'postOnly': True}
#             )
#             print(f"Placed sell order: {sell_amount} RED @ {sell_price}")

#             # 5. Monitor order execution
#             order_filled = False
#             for _ in range(15):  # Check for 30 seconds (15 * 2s)
#                 order_status = exchange.fetch_order(order['id'], symbol)
#                 if order_status['status'] == 'closed':
#                     print(f"Successfully sold {sell_amount} RED")
#                     order_filled = True
#                     break
            
#             if not order_filled:
#                 exchange.cancel_order(order['id'], symbol)
#                 print("Order not filled. Adjusting price...")
#                 target_price *= 0.995  # Reduce price by 0.5% each retry

#         except ccxt.InsufficientFunds:
#             print("No RED balance left")
#             return True
#         except Exception as e:
#             print(f"Error: {str(e)}")
#             time.sleep(1)


# Attempt to place orders
attempt_place_orders()
# sell_all_red_balance(target_price=1.1)