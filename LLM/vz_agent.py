from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"
google_api_key=os.getenv("GOOGLE_API_KEY")

# Load Gemini Flash 2.0
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))

# Prompt Template
prompt = PromptTemplate.from_template(
    "Classify the following user input into one of these: [finance, health, education, tech].\nInput: {text}\nCategory:"
)

# Chain: Input -> Prompt -> LLM -> Parser
chain = prompt | llm | StrOutputParser()

def classify_input(user_input):
    return chain.invoke({"text": user_input}).strip().lower()
