from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
from colorama import Fore, Style
import time

class BinanceAPIValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_api_credentials(self, api_key, api_secret, testnet=False):
        """
        Comprehensive validation of Binance API credentials and permissions
        Returns (bool, str) tuple: (is_valid, error_message)
        """
        try:
            client = Client(api_key, api_secret, testnet=testnet)
            
            # Step 1: Basic Connection Test
            server_time = client.get_server_time()
            if not server_time:
                return False, "Could not connect to Binance servers"
                
            # Step 2: Account Access Test
            account = client.get_account()
            if not account:
                return False, "Could not retrieve account information"
                
            # Step 3: Permission Checks
            permissions = self._check_required_permissions(client)
            if not permissions['success']:
                return False, permissions['message']
                
            return True, "API credentials validated successfully"
            
        except BinanceAPIException as e:
            error_message = self._handle_api_error(e)
            return False, error_message
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
            
    def _check_required_permissions(self, client):
        """
        Check if the API key has all required permissions
        """
        try:
            # Check 1: Read Permissions
            try:
                client.get_account()
            except BinanceAPIException as e:
                if e.code == -2015:
                    return {
                        'success': False,
                        'message': "API key lacks READ permissions"
                    }
                    
            # Check 2: Spot Trading Permissions
            try:
                # Just create a test order without sending
                client.create_test_order(
                    symbol='BTCUSDT',
                    side='BUY',
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity=0.001,
                    price=1000
                )
            except BinanceAPIException as e:
                if e.code == -2015:
                    return {
                        'success': False,
                        'message': "API key lacks SPOT TRADING permissions"
                    }
                    
            return {
                'success': True,
                'message': "All required permissions are present"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Error checking permissions: {str(e)}"
            }
            
    def _handle_api_error(self, error):
        """
        Handle specific Binance API errors with detailed messages
        """
        error_messages = {
            -2015: "Invalid API-key, IP, or permissions. Please check:\n"
                  "1. API key and secret are correct\n"
                  "2. IP whitelist settings in Binance\n"
                  "3. Required permissions are enabled:\n"
                  "   - Enable Reading\n"
                  "   - Enable Spot & Margin Trading",
            -2014: "API-key format invalid",
            -2013: "Order would trigger immediately",
            -2011: "Quantity too low for trade",
            -2010: "Insufficient funds",
            -1021: "Timestamp for this request was outside the recvWindow",
            -1013: "Invalid quantity",
            -1003: "Too many requests - please use WebSocket for live updates"
        }
        
        return error_messages.get(error.code, f"Unknown error: {error.message}")
        
    def verify_trading_permissions(self, client, symbol):
        """
        Verify specific trading permissions for a symbol
        """
        try:
            # Step 1: Check if trading is enabled
            exchange_info = client.get_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            
            if not symbol_info:
                return False, f"Symbol {symbol} not found"
                
            if symbol_info['status'] != 'TRADING':
                return False, f"Trading is not active for {symbol}"
                
            # Step 2: Check account status
            account_info = client.get_account()
            if not account_info['canTrade']:
                return False, "Account trading is disabled"
                
            return True, "Trading permissions verified"
            
        except BinanceAPIException as e:
            return False, self._handle_api_error(e)
        except Exception as e:
            return False, f"Error verifying trading permissions: {str(e)}"

    def suggest_fixes(self, error_code):
        """
        Provide specific fixes for common API issues
        """
        fixes = {
            -2015: [
                "1. Go to Binance API Management page",
                "2. Check if the API key is restricted to specific IPs",
                "3. Enable the following permissions:",
                "   - Enable Reading",
                "   - Enable Spot & Margin Trading",
                "4. If using Testnet, make sure to use testnet API credentials",
                "5. Verify the API key and secret are correctly copied"
            ],
            -2014: [
                "1. Regenerate the API key",
                "2. Make sure to copy the full API key without any spaces",
                "3. Check for special characters in the key"
            ],
            -1021: [
                "1. Synchronize your system time",
                "2. Increase the recvWindow parameter",
                "3. Check your internet connection stability"
            ]
        }
        return fixes.get(error_code, ["Please contact Binance support for assistance"])