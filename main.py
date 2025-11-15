import os

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from schemas import AgentResponse

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

# Create the ReAct agent usingthe new v1-alpha interface
agent = create_agent(
    llm,
    tools,
    response_format=AgentResponse,
)


def main():
    print("Hello from langchain-course!")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search for 3 job postings for ai engineer full remote in Argentina in Linkedin and summarize them with their source URLs.",
                }
            ]
        }
    )
    print(result["structured_response"])


if __name__ == "__main__":
    main()
