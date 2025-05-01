import pandas as pd
import numpy as np
import boto3
from io import StringIO

class StockMarketSimulation:
    def __init__(self, s3_bucket=None, s3_key=None):
        self.stock_names = ["TECH", "ENERGY", "RETAIL"]
        self.initial_prices = {
            "TECH": 150.0,
            "ENERGY": 75.0,
            "RETAIL": 50.0
        }
        
        # S3 configuration
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key or "stock_market_data.csv"
        
        # Initialize the first day
        self.current_day = 1
        self.cash = 10000.0
        self.holdings = {stock: 0 for stock in self.stock_names} # Dictionary to store the number of shares owned of each stock
        self.current_prices = self.initial_prices.copy() # Dictionary to store the current price of each stock
        self.sentiments = {stock: np.random.choice(["positive", "neutral", "negative"]) for stock in self.stock_names} # Dictionary to store the sentiment of each stock
        self.total_stocks_traded = 0
    
        self._initialize_dataframe()
    
    def _initialize_dataframe(self):
        """Initialize the pandas DataFrame."""
        data = {
            "Day": [self.current_day],
        }
    
        for stock in self.stock_names:
            data[f"{stock}_Price"] = [round(self.current_prices[stock], 2)]
            data[f"{stock}_Owned"] = [self.holdings[stock]]
            data[f"{stock}_Value"] = [round(self.holdings[stock] * self.current_prices[stock], 2)]
            data[f"{stock}_Sentiment"] = [self.sentiments[stock]]
            data[f"{stock}_Total_Bought"] = [0]
            data[f"{stock}_Daily_PnL"] = [0.0]
        
        data["Cash"] = [round(self.cash, 2)]
        data["Total_Invested"] = [round(sum(data[f"{stock}_Value"][0] for stock in self.stock_names), 2)]
        data["Portfolio_Value"] = [round(data["Cash"][0] + data["Total_Invested"][0], 2)]
        data["Total_Stocks_Traded"] = [self.total_stocks_traded]
        
        # Create DataFrame
        self.df = pd.DataFrame(data)
    
    def _update_prices(self):
        """Generate new prices for each stock."""
        # Update sentiments
        self._update_sentiments()
        
        for stock in self.stock_names:

            base_volatility = 0.07
            
            sentiment_effect = 0.0 # Neutral sentiment
            if self.sentiments[stock] == "positive":
                sentiment_effect = 0.02  
            elif self.sentiments[stock] == "negative":
                sentiment_effect = -0.02  
            
            # Generate price percent change, adjusting the center of the normal distribution based on sentiment
            percent_change = np.random.normal(loc=sentiment_effect, scale=base_volatility)
            
            # Apply the change
            self.current_prices[stock] *= (1 + percent_change)
            
            # Ensure price doesn't go below 1.0
            self.current_prices[stock] = max(1.0, self.current_prices[stock])
    
    def _update_sentiments(self):
        """Update market sentiment for each stock randomly."""
        for stock in self.stock_names:
            self.sentiments[stock] = np.random.choice(["positive", "neutral", "negative"])
    
    def advance_day(self):
        """Advance to the next day, update prices and all metrics."""
        self.current_day += 1
        
        # Generate new prices
        self._update_prices()
        
        # Create new row data
        new_row = {
            "Day": self.current_day,
            "Cash": round(self.cash, 2)
        }
        
        total_invested = 0
        
        # Update data for each stock
        for stock in self.stock_names:
            stock_value = self.holdings[stock] * self.current_prices[stock]
            total_invested += stock_value
            
            # Calculate daily profit/loss based on price change
            prev_price = self.df.iloc[-1][f"{stock}_Price"]
            price_change = self.current_prices[stock] - prev_price
            daily_pnl = price_change * self.holdings[stock]
            
            new_row[f"{stock}_Price"] = round(self.current_prices[stock], 2)
            new_row[f"{stock}_Owned"] = self.holdings[stock]
            new_row[f"{stock}_Value"] = round(stock_value, 2)
            new_row[f"{stock}_Sentiment"] = self.sentiments[stock]
            new_row[f"{stock}_Total_Bought"] = self.df.iloc[-1][f"{stock}_Total_Bought"]
            new_row[f"{stock}_Daily_PnL"] = round(daily_pnl, 2)

        new_row["Total_Invested"] = round(total_invested, 2)
        new_row["Portfolio_Value"] = round(self.cash + total_invested, 2)
        new_row["Total_Stocks_Traded"] = self.total_stocks_traded
        
        # Add new row to DataFrame
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save updated data
        self.save_to_csv()
        
        return self.get_current_state()
    
    def buy_stock(self, stock_name, quantity):
        """
        Buy a specified quantity of a stock.
        
        Args:
            stock_name: Name of the stock to buy
            quantity: Number of shares to buy
            
        Returns:
            dict: Result of the transaction
        """
        if stock_name not in self.stock_names:
            return {"success": False, "message": f"Stock {stock_name} not found"}
        
        cost = self.current_prices[stock_name] * quantity
        
        if cost > self.cash:
            return {"success": False, "message": "Insufficient funds"}
        
        # Update holdings and cash
        self.holdings[stock_name] += quantity
        self.cash -= cost
        
        # Update total stocks traded counter
        self.total_stocks_traded += quantity
        
        # Update the DataFrame
        idx = self.df.index[self.df["Day"] == self.current_day].tolist()[0]
        self.df.at[idx, f"{stock_name}_Owned"] = self.holdings[stock_name]
        self.df.at[idx, f"{stock_name}_Value"] = round(self.holdings[stock_name] * self.current_prices[stock_name], 2)
        self.df.at[idx, f"{stock_name}_Total_Bought"] = self.df.at[idx, f"{stock_name}_Total_Bought"] + quantity
        self.df.at[idx, "Cash"] = round(self.cash, 2)
        self.df.at[idx, "Total_Invested"] = round(sum(self.holdings[stock] * self.current_prices[stock] for stock in self.stock_names), 2)
        self.df.at[idx, "Portfolio_Value"] = round(self.cash + self.df.at[idx, "Total_Invested"], 2)
        self.df.at[idx, "Total_Stocks_Traded"] = self.total_stocks_traded
        
        # Save updated data
        self.save_to_csv()
        
        return {
            "success": True, 
            "message": f"Bought {quantity} shares of {stock_name} at ${self.current_prices[stock_name]:.2f} per share",
            "current_state": self.get_current_state()
        }
    
    def sell_stock(self, stock_name, quantity):
        """
        Sell a specified quantity of a stock.
        
        Args:
            stock_name: Name of the stock to sell
            quantity: Number of shares to sell
            
        Returns:
            dict: Result of the transaction
        """
        if stock_name not in self.stock_names:
            return {"success": False, "message": f"Stock {stock_name} not found"}
        
        if quantity > self.holdings[stock_name]:
            return {"success": False, "message": f"You only own {self.holdings[stock_name]} shares of {stock_name}"}
        
        # Calculate proceeds
        proceeds = self.current_prices[stock_name] * quantity
        
        # Update holdings and cash
        self.holdings[stock_name] -= quantity
        self.cash += proceeds
        
        # Update total stocks traded counter
        self.total_stocks_traded += quantity
        
        # Update the DataFrame
        idx = self.df.index[self.df["Day"] == self.current_day].tolist()[0]
        self.df.at[idx, f"{stock_name}_Owned"] = self.holdings[stock_name]
        self.df.at[idx, f"{stock_name}_Value"] = round(self.holdings[stock_name] * self.current_prices[stock_name], 2)
        self.df.at[idx, "Cash"] = round(self.cash, 2)
        self.df.at[idx, "Total_Invested"] = round(sum(self.holdings[stock] * self.current_prices[stock] for stock in self.stock_names), 2)
        self.df.at[idx, "Portfolio_Value"] = round(self.cash + self.df.at[idx, "Total_Invested"], 2)
        self.df.at[idx, "Total_Stocks_Traded"] = self.total_stocks_traded
        
        # Save updated data
        self.save_to_csv()
        
        return {
            "success": True, 
            "message": f"Sold {quantity} shares of {stock_name} at ${self.current_prices[stock_name]:.2f} per share",
            "current_state": self.get_current_state()
        }
    
    def get_current_state(self):
        """Get the current state of the market and portfolio."""
        current_data = self.df.iloc[-1].to_dict() # Get the last row of the DataFrame and convert it to a dictionary
        
        # Format data into a dictionary for output
        formatted_data = {
            "day": int(current_data["Day"]),
            "cash": round(float(current_data["Cash"]), 2),
            "total_invested": round(float(current_data["Total_Invested"]), 2),
            "portfolio_value": round(float(current_data["Portfolio_Value"]), 2),
            "stocks": {} # Dictionary to store the current state of each stock
        }
        
        for stock in self.stock_names:
            formatted_data["stocks"][stock] = {
                "price": round(float(current_data[f"{stock}_Price"]), 2),
                "owned": int(current_data[f"{stock}_Owned"]),
                "value": round(float(current_data[f"{stock}_Value"]), 2),
                "sentiment": current_data[f"{stock}_Sentiment"],
            }
        
        return formatted_data
    
    def save_to_csv(self):
        """Save the current DataFrame to a CSV file in S3."""
        if self.s3_bucket:
            try:
                # Initialize AWS S3 client
                s3_client = boto3.client('s3')
                
                # Convert DataFrame to CSV string stored in memory
                csv_buffer = StringIO() # StringIO class is a file-like object that allows you to read and write strings in memory
                self.df.to_csv(csv_buffer, index=False) # Convert the DataFrame to a CSV string to keep it in memory
                
                # Upload to S3
                s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=self.s3_key,
                    Body=csv_buffer.getvalue()
                )
                print(f"Data saved to S3: s3://{self.s3_bucket}/{self.s3_key}")
            except Exception as e:
                print(f"Error saving to S3: {e}")
            
    def reset_simulation(self):
        """Reset the simulation to its initial state."""
        self.current_day = 1
        self.cash = 10000.0
        self.holdings = {stock: 0 for stock in self.stock_names}
        self.current_prices = self.initial_prices.copy()
        self.sentiments = {stock: np.random.choice(["positive", "neutral", "negative"]) for stock in self.stock_names}
        self.total_stocks_traded = 0
        
        self._initialize_dataframe()
        
        self.save_to_csv()
        
        return {
            "success": True,
            "message": "Started a new simulation",
            "current_state": self.get_current_state()
        }


if __name__ == "__main__":
    # Initialize the simulation
    simulation = StockMarketSimulation()
    
    # Print the current state
    current_state = simulation.get_current_state()
    print("Welcome to the Stock Market Simulation!")
    print("Current State:", current_state)
