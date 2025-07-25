from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
 
load_dotenv()
 
model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
 
result = model.invoke("Share few words about india and its culture")
 
print(result.content)