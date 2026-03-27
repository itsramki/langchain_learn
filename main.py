import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_ollama import ChatOllama

load_dotenv()

tools = []

def main():
    #print("Hello from langchain-course!")
    #print(os.environ.get("MISTRAL_API_KEY"))

    information = """ Sachin Tendulkar """

    summary_template = f""" I want you to create :
                        1. A short summary
                        2. two interesting facts about {information}
                    """

    print(summary_template)
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatMistralAI(
       model = "mistral-tiny-latest",
       temperature = 0.0
    )

    # llm = ChatOllama(
    #     model = "gemma3:270m",
    #     temperature = 0.0
    # )

    chain = summary_prompt_template | llm

    response = chain.invoke(input = {"information": information})

    print(response.content)


if __name__ == "__main__":
    main()
