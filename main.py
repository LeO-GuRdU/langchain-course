import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

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

class Source(BaseModel):
    """
    Schema for a source in the search results
    """
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """
    Schema for the agent response
    """
    answer: str = Field(description="The agent answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

groq_api_key = os.getenv("GROQ_API_KEY")
# llm = ChatGroq(
#     model="openai/gpt-oss-20b",
#     temperature=0,
#     api_key=groq_api_key,
# )

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(
        content="Busca ofertas de empleo en linkedin para ingeniero IA Langchain en USA, que acepte full remoto y resume los  3 mejores resultados"
        )})
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
