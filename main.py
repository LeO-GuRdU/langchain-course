import os

from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

google_api_key = os.getenv("GOOGLE_API_KEY")
tools = [TavilySearch()]
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=google_api_key,
)

# llm = ChatOllama(
#     model="mistral:latest",
#     temperature=0,
# )
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt=react_prompt)
agent_executor = AgentExecutor.from_agent_and_tools(agent, tools, verbose=True)

def main():
    print("Hello from langchain-course!")

    response = agent_executor.invoke(input={"input": "Search for 3 job postings for ai engineer in Argentina and summarize them."})
    print("Respuesta del agente:", response)


if __name__ == "__main__":
    main()
