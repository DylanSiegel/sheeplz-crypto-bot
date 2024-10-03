# main.py

import asyncio
from models.utils.data_ingestion import DataIngestion
from models.gmn.gmn import GraphMetanetwork
from models.agents.agent import TradingAgent

async def main():
    # Initialize components
    data_ingestion = DataIngestion()
    gmn = GraphMetanetwork()
    agent_1m = TradingAgent(time_frame="1m")
    agent_1h = TradingAgent(time_frame="1h")
    agents = [agent_1m, agent_1h]

    # Initialize GMN nodes
    gmn.initialize_nodes(
        time_frames=["1m", "5m", "1h", "1d"],
        indicators=["price", "volume", "rsi", "macd", "fibonacci"]
    )

    # Start data ingestion and agent decision-making
    await asyncio.gather(
        data_ingestion.connect(),
        agent_loop(agents, gmn)
    )

async def agent_loop(agents, gmn):
    while True:
        market_data = gmn.get_latest_data()
        for agent in agents:
            agent.make_decision(market_data)
        await asyncio.sleep(1)  # Adjust sleep time as needed

if __name__ == "__main__":
    asyncio.run(main())
