import time
import pandas as pd
import logging
import json
import os
import numpy as numpy
from binance.client import Client
from decimal import Decimal, ROUND_DOWN

class BinanceHandler:

    def __init__(self, api_key, api_secret, symbol):
        self.client = Client(api_key, api_secret,tld='com')
        self.symbol = symbol
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(filename='trading_bot.log', level=logging.INFO,
                            format='%(asctime)s:%(levelname)s:%(message)s')
        
    def get_file_name(self, start_time, end_time):
        # Create a unique file name for each time range
        return f"historical_data/{self.symbol}_{start_time}_{end_time}.json"

    def save_data_to_file(self, filename, data):
        with open(filename, 'w') as file:
            json.dump(data, file)

    def load_data_from_file(self, filename):
        with open(filename, 'r') as file:
            return json.load(file)

    def get_historical_data(self, start_time, end_time):
        filename = self.get_file_name(start_time, end_time)

        # Check if the data is already available locally
        if os.path.exists(filename):
            return self.load_data_from_file(filename)

        dirname = os.path.abspath(os.path.join(filename, os.pardir))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Fetch historical minute klines for the last 'x' minutes.
        klines = self.client.get_historical_klines(self.symbol, Client.KLINE_INTERVAL_1MINUTE, start_time, end_time)

        # Save the data locally for future use
        self.save_data_to_file(filename, klines)

        return klines   
 
    def get_account_balance(self):
        # Fetch account details
        account_details = self.client.get_account()
        # Extract balances for BTC and base currency (assuming USDT for simplicity)
        btc_balance = next((item for item in account_details['balances'] if item["asset"] == "BTC"), None)
        usdt_balance = next((item for item in account_details['balances'] if item["asset"] == "EUR"), None)
        return float(btc_balance['free']), float(usdt_balance['free'])


    def get_trade_fee(self, symbol):
        """Fetch the trade fee for the specified symbol."""
        trade_fee = self.client.get_trade_fee(symbol=symbol)
        # Extract the maker or taker fee rate (as appropriate for your order type) from the response
        fee_rate = float(trade_fee[0]['takerCommission'])  
        return fee_rate
    
    def adjust_quantity(self, quantity):
        # Adjust the quantity to the allowed precision
        step_size_str = "{:.8f}".format(self.get_lot_size()).rstrip('0')
        quantity_str = "{:0.0{}f}".format(quantity, len(step_size_str.split('.')[-1]))
        adjusted_quantity = Decimal(quantity_str).quantize(Decimal(step_size_str), rounding=ROUND_DOWN)
        return adjusted_quantity

    def get_lot_size(self):
        info = self.client.get_symbol_info(self.symbol)  # Replace with your symbol

        # Find the 'LOT_SIZE' filter
        lot_size_filter = next((f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            return float(lot_size_filter['stepSize'])
        else:
            raise ValueError("LOT_SIZE filter not found.")
        
    def get_max_quantity_buy(self):
        _, eur_balance = self.get_account_balance()
        current_price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])
        trade_fee_rate = self.get_trade_fee(self.symbol)  # Fetch the trade fee rate
        eur_balance_after_fee = eur_balance / (1 + trade_fee_rate)
        print(eur_balance_after_fee)
        quantity_to_buy = eur_balance_after_fee / current_price
        print(quantity_to_buy)
        adjusted_quantity = self.adjust_quantity(quantity_to_buy) 
        return adjusted_quantity
    
    def get_max_quantity_sell(self):
        btc_balance, _ = self.get_account_balance()
        trade_fee_rate = self.get_trade_fee(self.symbol)  # Fetch the trade fee rate
        btc_balance_after_fee = btc_balance * (1 - trade_fee_rate)
        quantity_to_sell = btc_balance_after_fee
        adjusted_quantity = self.adjust_quantity(quantity_to_sell) 
        return adjusted_quantity 
    
    def execute_buy(self, quantity):
        try:
            order = self.client.order_market_buy(symbol=self.symbol, quantity=quantity)
            logging.info(f"Buy order executed: {order}")
        except Exception as e:
            logging.error(f"An error during buy order occurred: {e}")


    def execute_sell(self, quantity):
        try: 
            order = self.client.order_market_sell(symbol=self.symbol, quantity=quantity)
            logging.info(f"Sell order executed: {order}")
        except Exception as e:
            logging.error(f"An error during sell order occurred: {e}")
