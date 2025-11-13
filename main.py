import os

from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

groq_api_key = os.getenv("GROQ_API_KEY")
tools = [TavilySearch()]

def main():
    print("Hello from langchain-course!")

    # llm = ChatGroq(
    #     model="openai/gpt-oss-20b",
    #     temperature=0,
    #     api_key=groq_api_key
    # )

    llm = ChatOllama(
        model="mistral:latest",
        temperature=0,
    )

if __name__ == "__main__":
    main()
