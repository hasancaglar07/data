from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",
    temperature=0.7
)

response = llm.invoke("Merhaba, tasavvuf nedir?")
print(response.content)