# Agentic Investing with LLM Agents and MCP
<p align="center">
<img src="https://www.livemint.com/lm-img/img/2025/03/31/600x338/g1c49305ef246b25d62f_1743440565533_1743440565716.jpg" style="width: 40%;">
</p>

## Table of Contents
<table>
  <tr>
    <td><a href="#introduction">Introduction</a><br>
    <a href="#stock-market-simulation">Stock Market Simulation</a><br>
    <a href="#mcp-server">MCP Server</a><br>
    <a href="#hosting-on-aws">Hosting on AWS</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;<a href="#docker">Docker</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;<a href="#ec2">EC2</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;<a href="#s3">S3</a><br>
    <a href="#agents">Agents</a><br>
    <a href="#powerbi">Power BI Dashboard</a><br>
    <a href="#conclusion">Conclusion</a></td>
  </tr>
</table>

## Introduction
<a name="introduction"></a>

For this project, I built a simplified stock market simulation, three types of Large Language Model (LLM) agents using the OpenAI Agents SDK, and a Model Context Protocol (MCP) server hosted on Amazon Web Services (AWS) that provided the tools the agents used to interact with the simulation. To evaluate the agent performance, tool use, and decision making, I designed a Power BI dashboard using the simulation data to visualize key metrics. All together, it proved to be a comprehensive platform and testbed for exploring LLM agents and MCP servers.

The increasing capabilities and decreasing costs of LLMs are fueling rapid growth in agentic systems across various industries. Businesses are increasingly leveraging agentic systems to boost efficiency, cut operational expenses, and innovate new services. As a result, the ability to effectively develop, test, and evaluate these agentic solutions is becoming a critical skill.

**The core components of this project include:**
*   A Stock Market Simulation built with Python that models fictional stocks with fluctuating prices based on randomized daily sentiment.
*   An MCP Server using the FastMCP library that acts as an interface, exposing simulation functions (like buying/selling stock, checking state) as tools for LLM agents.
*   Three distinct LLM Agents (Interactive, Human-in-the-Loop, Autonomous) developed using the OpenAI Agents SDK and leveraging gpt-4o-mini, each showcasing different levels of user interaction and autonomy.
*   Cloud Infrastructure utilizing AWS. The MCP server and simulation are containerized using Docker and hosted on an EC2 instance for accessibility. Simulation data is stored in an S3 bucket as a CSV file.
*   A Power BI Dashboard that reads the CSV file, providing visual analytics and key performance indicators (KPIs) to evaluate agent performance.

This setup would allow for rapid iteration over agent configurations and tools. The resulting simulation data, visualized in Power BI, offers valuable feedback for evaluating the effectiveness of agent instructions, tool design, and workflow.

## Stock Market Simulation
<a name="stock-market-simulation"></a>
**Code:** **[simulation.py](Portfolio/Investing%20Game/docker/simulation.py)**

The core of the simulation is the StockMarketSimulation class. This class manages the state of the agents' portfolio and the simulation's three fictional stocks: TECH, ENERGY, and RETAIL. Each stock is randomly assigned a daily sentiment (positive, neutral, or negative) which influences the stock price update. The agents start Day 1 with an initial cash balance of $10,000 and it is their goal to help maximize the portfolio value by investing based on the market sentiments of the stocks.

The daily stock price update introduces randomness while incorporating market sentiment. The updated price is calculated based on the current price using the following formula, ensuring the price never drops below $1.00:

updated_price = max(1.0, current_price * (1 + percent_change))

Where the percent change is determined by generating a random number from a normal distribution using numpy.random.normal. This function generates random values centered around a mean (loc) with a specific standard deviation (scale):

percent_change = numpy.random.normal(loc=sentiment_effect, scale=base_volatility)

-   `base_volatility`: Is set to 0.07
-   `sentiment_effect`: Adjusts the mean of the random price change based on the sentiment:
    -   Positive sentiment: 0.02
    -   Neutral sentiment: 0.0
    -   Negative sentiment: -0.02

This approach means that while positive sentiment increases the likelihood of a price increase, the inherent volatility still allows for the possibility of a price decrease and vice versa.

**Data:**

The simulation state is maintained using a pandas DataFrame which is saved to a CSV file for later analysis. Each row of the DataFrame represents a day of the simulation. Metrics were chosen based on their usefulness for agent decision making and performance evaluation.

Key metrics stored include:
-   `Day`: The current simulation day.
-   `{Stock}_Price`: Price of the stock.
-   `{Stock}_Owned`: Number of shares owned.
-   `{Stock}_Value`: Total value of owned shares for that stock.
-   `{Stock}_Sentiment`: Market sentiment for the stock.
-   `{Stock}_Total_Bought`: Cumulative shares bought (used for tracking activity, not PnL).
-   `{Stock}_Daily_PnL`: Profit or loss for the stock based only on the price change from the previous day.
-   `Cash`: Available cash balance.
-   `Total_Invested`: Total value of all stock holdings.
-   `Portfolio_Value`: Total value of cash + invested assets.
-   `Total_Stocks_Traded`: Cumulative count of shares bought or sold across all transactions.

**Core Functions:**  

There are five functions that are used to interact with the simulation. Each of these functions is provided as tools to the LLM agents via the MCP server.

-   `advance_day()`: Moves the simulation to the next day, updates prices and sentiments, calculates daily profit/loss, and records the new state in the DataFrame.
-   `buy_stock(stock_name, quantity)`: Allows purchasing a specified quantity of a stock if sufficient cash is available. Updates holdings, cash, and relevant metrics.
-   `sell_stock(stock_name, quantity)`: Allows selling a specified quantity of owned stock. Updates holdings, cash, and relevant metrics.
-   `get_current_state()`: Returns a dictionary summarizing the current simulation state, including day, cash, portfolio value, and details for each stock (price, owned shares, value, sentiment).
-   `reset_simulation()`: Resets the simulation back to its initial Day 1 state.

Importantly, these functions return statements that contain success status, messages, and updated simulation states. This feedback is essential for the LLM agents to understand the outcome of their actions after each tool use and to help them plan subsequent steps. To minimize token usage and context size, this feedback is formatted for LLM consumption using a python dictionary format which humans might find difficult to read. While testing, I found that the LLM agents were effective at reformatting the data into human-readable formats for the user.

## MCP Server
<a name="mcp-server"></a>
**Code:** **[mcp_server.py](Portfolio/Investing%20Game/docker/mcp_server.py)**  
**Implementation Resources:** [MCP User Guide](https://modelcontextprotocol.io/introduction) and [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

MCP servers allow for services to provide context to LLMs in the form of tools, prompts, and resources. This implementation uses the FastMCP library to create a "HTTP over SSE" server that runs remotely. To set up an SSE server using FastMCP, run the FastMCP's `run_sse_async()` method using asyncio.run(). 

I highly recommend using the developer tool "MCP Inspector," which can be run with `mcp dev server.py`. It allows you to test the tools, essentially allowing you to interact with the server as the LLM would, which is useful for both development and debugging.

<p align="center">
<img src="https://raw.githubusercontent.com/MattPickard/Project_Portfolio/refs/heads/main/Images/mcp_inspector.png" style="width: 100%;">
</p>

**Tools:** 

The MCP server automatically provides a list of available tools to the LLM agent when it communicates with the server. Each tool is a function defined using the `@mcp.tool()` decorator. The agents are exposed to the docstrings of each tool, so tool-specific instructions can be provided in the docstring. For example, I included instructions about stock sentiment in the `buy_stock` tool's docstring as follows:

```python
@mcp.tool()
def buy_stock(stock_name, quantity):
    """
    Buy a specified quantity of a stock. Buying stocks with positive sentiment should be prioritized.
    
    Args:
        stock_name: Name of the stock to buy
        quantity: Number of shares to buy
    """
```

## Hosting on AWS
<a name="hosting-on-aws"></a>

To host the MCP server on AWS, I used Docker for containerization, EC2 for compute, and S3 for data storage using an AWS free-tier account.

### Docker
<a name="docker"></a>
**Files:** **[Dockerfile](Portfolio/Investing%20Game/docker/Dockerfile)**, **[requirements.txt](Portfolio/Investing%20Game/docker/requirements.txt)**

Docker was used to package the MCP server, stock market simulation, and dependencies into a container. This approach allows for consistent behavior across different environments. The Dockerfile defines the steps to build this container image:
-   Specifies the base image as python:3.10-slim since the FastMCP library requires Python 3.10 or higher.
-   Installs necessary system packages (gcc, python3-dev) and Python libraries specified in `requirements.txt` (which includes pandas, numpy, boto3, fastmcp).
-   Copies the application code into the container's working directory.
-   Exposes the container on port 8080, which the MCP server listens on.
-   Sets the container to run the mcp_server.py script when it starts.

### EC2
<a name="ec2"></a>

An Amazon EC2 instance was used to host the Docker container. The setup process involved the following steps:
1.  Launching an EC2 instance using the Amazon Linux 2023 AMI.
2.  Installing Docker on the instance.
3.  Uploading the project files (Dockerfile, requirements.txt, Python scripts) to the instance using Secure Copy Protocol (SCP).
4.  Building the Docker image on the EC2 instance using the command `docker build -t investing-sim .`.
5.  Configuring the instance's Security Group to allow inbound TCP traffic on port 8080.
6.  Running the Docker container:
    ```bash
    docker run -d -p 8080:8080 \
      -e PORT=8080 \
      -e S3_BUCKET="investing-sim-data" \
      -e S3_KEY="stock_market_data.csv" \
      --name investing-sim \
      investing-sim
    ```

### S3
<a name="s3"></a>

To store the simulation data for later analysis, I utilized Amazon S3. After creating an S3 bucket, I configured the save_to_csv method in simulation.py to save the DataFrame to an in-memory text buffer using io.StringIO in CSV format. The buffer is then uploaded directly to the S3 bucket using Amazon's boto3 client's put_object method. By uploading to S3, the simulation's history is saved even if the EC2 instance or Docker container is stopped or restarted. For larger scale applications, it would be more efficient to use a relational database with Amazon RDS instead of saving to an S3 bucket.

## Agents
<a name="agents"></a>
**Code:** **[agents.py](Portfolio/Investing%20Game/docker/agents.py)**  
**Implementation Resources:** [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)

Three different agents were developed using the OpenAI Agents SDK to interact with the simulation via the MCP server. To explore the capabilities of smaller models and ensure efficiency, gpt-4o-mini was used as the base model for each agent. On startup, the user is prompted to choose between the three agent types. 

**Agent Types:**
1.  **Interactive Investment Advisor:** Acts as a conversational assistant, explaining its reasoning and guiding the user through investment decisions and actions.
2.  **Human-in-the-Loop Agent:** Proactively proposes specific buy/sell actions, explains its rationale, then *waits for user confirmation* before executing trades.
3.  **Fully Autonomous Agent:** Operates completely independently to maximize portfolio value over a set period (20 days). It checks the state, makes decisions based on sentiment, executes trades, and advances the day automatically until the target day is reached. It provides a summary at the end.

**Configuration:**
Key components of the agents include:
-   `Agent` Class initialized with `instructions`: `instructions` act as the system prompt defining the agent's role, goals, and operational guidelines, and given as context at each turn.
-   A connection to the MCP server using the MCPServerSse instance.
-   `Runner.run` Method: Used to execute the agent with a given input (initial prompt or conversation history).
-   Interaction loops to manage the turn-by-turn conversation for the interactive advisor and human-in-the-loop agents.
    -   Conversation history is maintained by passing the previous `result.to_input_list()` along with the new user message to subsequent `Runner.run` calls.
-   An `Agent` class parameter `max_turns` that stops the agent and returns an error if the agent takes more than `max_turns` turns. This is a guardrail to prevent agents from entering infinite loops. Be aware that this parameter is set to 10 by default. The autonomous agent had it set to 60, allowing for an average of 3 turns per day (each tool call is considered a turn).

**OpenAI Traces:**

It's worth noting that OpenAI provides [built-in tracing functionality](https://openai.github.io/openai-agents-python/tracing/) as part of their Agents SDK. The Traces dashboard on their website breaks down each action taken by each agent. This can be useful when developing both single-agent and more complex multi-agent systems.

<p align="center">
<img src="https://raw.githubusercontent.com/MattPickard/Project_Portfolio/refs/heads/main/Images/traces.png" style="width: 100%;">
</p>

## Power BI Dashboard
<a name="powerbi"></a>

<p align="center">
<img src="https://raw.githubusercontent.com/MattPickard/Project_Portfolio/refs/heads/main/Images/powerbi_dashboard.png" style="width: 100%;">
</p>

I used a Power BI dashboard to create a comprehensive visual summary of the stock market simulation and the agent's performance. The dashboard uses the simulation's CSV file to display visualizations and key metrics that help a potential developer save time evaluating and iterating on agent performance. The dashboard includes the following:

-   **Stock Prices (Time Series):** Useful for comparing stock trends to portfolio performance and types of stocks bought.

-   **Daily Performance by Owned Stocks (Grouped Column):** Reveals which sectors contributed most to gains or losses on any given day and creates a rough timeline of which stocks were owned when.

-   **Portfolio Value (Time Series + Key Performance Indicator):** The value the agents are attempting to maximize.

-   **Stocks Bought (Donut):** Visualizes the proportion of each type of stock bought, along with the absolute number and percentage, giving insights into the agent's investment strategy.

-   **Stocks Held or Bought by Sentiment (Table):** Reveals to what extent the agent is properly incorporating market sentiment into its decision-making process. The DAX formula for positive sentiment count is as follows, the same formula was applied for neutral and negative sentiment counts:
```
        count_positive = 
        COUNTROWS(
            FILTER(
                stock_market_data,
                (stock_market_data[ENERGY_Owned] <> 0 && stock_market_data[ENERGY_Sentiment] = "positive") ||
                (stock_market_data[RETAIL_Owned] <> 0 && stock_market_data[RETAIL_Sentiment] = "positive") ||
                (stock_market_data[TECH_Owned] <> 0 && stock_market_data[TECH_Sentiment] = "positive")
            )
        )
```
-   **Avg Cash Held (Key Performance Indicator):** Calculates the average cash balance held by the agent at the end of each simulation day. Importantly, days where all three stocks had negative sentiment are excluded from this average. This exclusion acknowledges that holding cash is the most reasonable strategy on such days. A lower average indicates whether the agent is actively investing available funds as instructed, rather than holding excessive cash unnecessarily. The DAX formula is as follows:
```
        AverageCashHeld = 
        AVERAGEX(
            FILTER(
                stock_market_data,
                NOT(stock_market_data[ENERGY_Sentiment] = "negative") &&
                NOT(stock_market_data[RETAIL_Sentiment] = "negative") &&
                NOT(stock_market_data[TECH_Sentiment] = "negative")
            ),
            stock_market_data[Cash]
        )
```
## Conclusion
<a name="conclusion"></a>

By combining a stock market simulation, an MCP server hosted on AWS (EC2, S3), various LLM agents, and a Power BI dashboard, this project provides a platform for developing, testing, and evaluating agentic workflows and tools. The setup enables rapid iteration on agent design while providing clear insights into agent behavior and effectiveness. Future enhancements could include generating synthetic news events to test sentiment analysis capabilities, integrating real-time data feeds, or adapting the system to work with real-world stocks using libraries like Alpaca-py or yfinance.

Ultimately, this project provided me with experience in tool design, MCP server implementation, agent creation, and cloud deployment — skills useful for building agentic AI applications. As LLM capabilities improve and costs decrease, wide-scale adoption of agentic technologies feels inevitable. In a future where human-AI collaboration is more commonplace, the ability to connect agents with business services will become increasingly valuable. MCP shows promise as a potential standard for how businesses control and facilitate agent interactions with their services.
