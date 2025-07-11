## Working code
#pip install fastapi langchain requests uvicorn

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
import requests
from vz_case_followup5_restendpoint import generate_intent
from pydantic import BaseModel

# Request model
class MessageRequest(BaseModel):
    message: str


# Function that calls the FastAPI endpoint
def call_cjcm_endpoint(name: str) -> str:
    response = requests.get(f"http://127.0.0.1:8000/cjcm/retrieve/message")
    #print("Response from CJCM endpoint:=======>", response.json)
    return response.json()

def call_ccai_endpoint(name: str) -> str:
    response = requests.get(f"http://127.0.0.1:8000/ccai/retrieve/message")
    #print("Response from CCAI endpoint:=======>", response.json)
    return response.json()

def call_genai_endpoint(name: str) -> str:
    response = requests.get(f"http://localhost:8000/case-followup")
    #print("Response from Gen Ai Model :=======>", response.json)
    return response.json()

# Create a LangChain Tool
cjcm_messages_pull_agent = Tool(
    name="CJCM_Messages_Pull_Agent",
    func=call_cjcm_endpoint,
    description="This Agent Calls CJCM endpoint to get Case Number"
)

# Create a LangChain Tool
ccai_messages_pull_agent = Tool(
    name="CCAI_Messages_Pull_Agent",
    func=call_ccai_endpoint,
    description="This Agent Calls CCAI endpoint to get Live Message between Customer and Agent"
)

# Create a LangChain Tool
create_case_LLM_tool = Tool(
    name="CreateCaseGenAiModelAgent",
    func=call_genai_endpoint,
    description="This Agent Calls Gen Ai Model endpoint to get generated intent from the customer message"
)

def rule_based_decide_ccai_call(name: str) -> str:
    output_msg = ""
    cjcm_agent_response = cjcm_messages_pull_agent.run("")
    print("CJCM Agent Response: ", cjcm_agent_response)
    for key, value in cjcm_agent_response.items():
        #print(f"Key: {key}, Value: {value}")
        if(value=="message1" and "ticket number" in value or "case details available" in value):
            #print("Message1======>", key, value)
            output_msg = value
        else:
            #call Generative AI model to generate response
            print("Message2======>", "Going to CCAI api call, invoking Model, generating response from Gemini AI")
            # Call the CCAI endpoint
            ccai_api_agent_response = ccai_messages_pull_agent.run("")
            print("CCAI Agent Response: ", ccai_api_agent_response)    
            msg = MessageRequest(message=ccai_api_agent_response.get("msg_value"))
            gemini_ai_response = generate_intent(msg)
            print("Success::::Gemini AI Response: ", gemini_ai_response)
            output_msg = gemini_ai_response
    return output_msg


if __name__ == "__main__":
    #cjcm_api_response = cjcm_tool.run("")
    #print("cjcm_api_response=====", cjcm_api_response)
    #ccai_api_response = ccai_tool.run("")
    #print("ccai_api_response=====", ccai_api_response)
    dashboard_output = rule_based_decide_ccai_call("test")
    print("Dashboard Output: ", dashboard_output)
