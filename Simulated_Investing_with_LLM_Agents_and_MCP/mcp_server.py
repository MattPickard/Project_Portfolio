from mcp.server.fastmcp import FastMCP
import simulation
import asyncio
import os

# Create an MCP server instance named "InvestingSim"
PORT = int(os.environ.get("PORT"))
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_KEY = os.environ.get("S3_KEY")

# Create an MCP server instance named "InvestingSim"
mcp = FastMCP("InvestingSim", port=PORT)

# Initialize the simulation
sim = simulation.StockMarketSimulation(
    s3_bucket=S3_BUCKET,
    s3_key=S3_KEY
)

@mcp.tool()
def advance_day():
    """Advance to the next day, update prices and all metrics."""
    return sim.advance_day()

@mcp.tool()
def buy_stock(stock_name, quantity):
    """
    Buy a specified quantity of a stock. Buying stocks with positive sentiment should be prioritized.
    
    Args:
        stock_name: Name of the stock to buy
        quantity: Number of shares to buy
    """
    return sim.buy_stock(stock_name.upper(), int(quantity))

@mcp.tool()
def sell_stock(stock_name, quantity):
    """
    Sell a specified quantity of a stock. Selling stocks with negative sentiment should be prioritized.
    
    Args:
        stock_name: Name of the stock to sell
        quantity: Number of shares to sell
    """
    return sim.sell_stock(stock_name.upper(), int(quantity))

@mcp.tool()
def get_current_state():
    """Get the current state of the simulation."""
    return sim.get_current_state()

@mcp.tool()
def reset_simulation():
    """Reset the simulation to its initial state. Only use this tool if the user asks you to reset the simulation."""
    return sim.reset_simulation()

if __name__ == "__main__":
    asyncio.run(mcp.run_sse_async())
