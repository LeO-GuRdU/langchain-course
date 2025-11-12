import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq




load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


def main():
    print("Hello from langchain-course!")
    information = """
Elon Reeve Musk[b] (born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2021; as of October 2025, Forbes estimates his net worth to be around $500 billion.

Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; his Canadian citizenship is congenital, his mother having been born there. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.
    """
    sumary_template = """
    given the {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=sumary_template,
    )

    # llm = ChatGroq(
    #     model="openai/gpt-oss-20b",
    #     temperature=0,
    #     api_key=groq_api_key
    # )

    llm = ChatOllama(
        model="gemma3:270m",
        temperature=0,
    )

    chain = summary_prompt_template | llm
    response = chain.invoke(
        input={
            "information": information,
        }
    )
    print("Response: ",response.content)


if __name__ == "__main__":
    main()
