import asyncio
from agents import Agent, Runner
from agents.mcp.server import MCPServerSse
import os
from dotenv import load_dotenv

# Set up the MCP server connection
async def setup_mcp_server():
    server = MCPServerSse(cache_tools_list=True,
        params = {"url": "http://_/sse"} #http://ip:port/sse
    )
    await server.connect()
    return server


async def run_interactive_agent():
    print("Starting interactive investment advisor...")
    
    # Connect to the MCP server
    mcp_server = await setup_mcp_server()
    
    # Create the interactive agent
    interactive_agent = Agent(
        name="Investment Advisor",
        instructions="""
        You are a investment advisor helping the user make decisions and invest in a stock market simulation.
        You need to guide the user through the process of investing. 
        Explain to the user what you would do and why at each step.
        """,
        model="gpt-4o-mini",
        mcp_servers=[mcp_server]
    )
    
    # Start the interactive session
    print("Type 'exit' to end the session")
    
    # Initial prompt fed to the agent
    result = await Runner.run(interactive_agent, "What do you think I should do?")
    print(f"Advisor: {result.final_output}")
    
    # Interactive loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        # Run the agent with the user's input
        result = await Runner.run(
            interactive_agent,
            result.to_input_list() + [{"role": "user", "content": user_input}]
        )
        
        print(f"Advisor: {result.final_output}")
    
    await mcp_server.cleanup()

async def run_human_in_loop_agent():
    print("Starting human-in-the-loop investment agent...")
    
    # Connect to the MCP server
    mcp_server = await setup_mcp_server()
    
    # Create the human-in-the-loop agent
    human_in_loop_agent = Agent(
        name="Human-in-the-Loop Investment Agent",
        instructions="""
        You are an investment agent that proposes decisions for the user to confirm.
        
        For each day:
        1. Check the current state of the portfolio and market
        2. Make smart investment decisions based on sentiment, prioritize investing in stocks with positive sentiment to maximize returns. You should sell non-positive sentiment stocks to invest in positive sentiment stocks.
        3. Buy/sell stocks strategically each day, selling if negative sentiment, buying if positive sentiment. You can make multiple trades a day. 
        4. Invest all cash into stocks unless all three stocks show negative sentiment for that day
        5. Propose all of the actions you will take for the day with clear rationale
        6. Wait for user confirmation before executing trades
        7. After user confirms, execute the trades and advance to the next day
        
        Always explain your reasoning clearly so the user can make informed decisions.
        """,
        model="gpt-4o-mini",
        mcp_servers=[mcp_server]
    )
    
    # Start the session with initial instruction
    print("Type 'exit' to end the session")

    # Initial prompt fed to the agent
    result = await Runner.run(
        human_in_loop_agent, 
        "I want you to help me maximize my portfolio value over 20 days. For each day, analyze the market, propose specific investment actions with your rationale, and wait for my confirmation before executing them."
    )
    print(f"Agent: {result.final_output}")
    
    # Interactive loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        # Run the agent with the user's input
        result = await Runner.run(
            human_in_loop_agent,
            result.to_input_list() + [{"role": "user", "content": user_input}]
        )
        
        print(f"Agent: {result.final_output}")
    
    await mcp_server.cleanup()

async def run_autonomous_agent():
    print("Starting autonomous investment agent...")
    
    # Connect to the MCP server
    mcp_server = await setup_mcp_server()
    
    # Create the autonomous agent
    autonomous_agent = Agent(
        name="Autonomous Investment Agent",
        instructions="""
        You are an autonomous investment agent whose goal is to maximize portfolio value over 20 days.
        
        Follow these steps for each day:
        1. Check the current state of the portfolio and market
        2. Make smart investment decisions based on sentiment, prioritize investing in stocks with positive sentiment to maximize returns. You should sell non-positive sentiment stocks to invest in positive sentiment stocks.
        3. Buy/sell stocks strategically each day, selling if negative sentiment, buying if positive sentiment. You can make multiple trades a day. 
        4. Invest all cash into stocks unless all three stocks show negative sentiment for that day
        5. Advance to the next day, repeating until day 20
        6. Stop at day 20, if it is over day 20, stop the process
        
        When finished, provide a short summary of the steps you took and your performance.
        """,
        model="gpt-4o-mini",
        mcp_servers=[mcp_server]
    )
    
    # Run the agent with initial instruction        
    result = await Runner.run(
        autonomous_agent, 
        "I want you to maximize the value of my portfolio over 20 days. Start by checking the current state, then make investment decisions for 20 consecutive days. Stop at day 20.",
        max_turns=60
    )
    print(result.final_output)
    
    await mcp_server.cleanup()



async def main():
    print("Investment Simulation Agents:")
    print("1. Run investment advisor agent")
    print("2. Run human-in-the-loop agent")
    print("3. Run fully autonomous agent (20 days)")
    
    choice = input("Select mode (1, 2, or 3): ")
    
    if choice == "1":
        await run_interactive_agent()
    elif choice == "2":
        await run_human_in_loop_agent()
    elif choice == "3":
        await run_autonomous_agent()
    else:
        print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    # Load OpenAI API key
    load_dotenv() 
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Run the main function
    asyncio.run(main())
