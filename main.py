import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()

# Uncomment the following lines to enable the custom Tavily search tool
# from tavily import TavilyClient


# tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over the internet
#     Args:
#         query: search query
#     Returns:
#         search results
#     """
#     print(f"Searching for: {query}")
#     return tavily.search(query=query)

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=groq_api_key
)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(
        content="Busca ofertas de empleo en linkedin para desarrollador Langchain en USA, que acepte full remoto y resume los resultados."
        )})
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
