from xmlrpc.client import boolean
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import random

app = FastAPI(debug=True)

# In-memory store (temporary; resets when app restarts)
#temp_store = {}
# In-memory store (key → list of objects)
cjcm_store: Dict[str, str] = {}

# Input model
class Message(BaseModel):
    msg_key: str
    msg_value: str
    #read: boolean  # Default value for read status

@app.post("/cjcm/store/")
def store_temp(messages: list[Message]):
    for message in messages:
            cjcm_store[message.msg_key] = message.msg_value
    return {"message": f"Stored {len(messages)} messages "}

@app.get("/cjcm/retrieve/{msg_key}")
def retrieve_temp(msg_key: str):
    #print("message1==>", cjcm_store["message1"]);
    #print("message2==>", cjcm_store["message2"]);

    num = random.randint(1, 2)  # random number between 1 and 3
    print(num)
    msg_key = msg_key+str(num)
    print(msg_key)
    if msg_key not in cjcm_store:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"msg_key": msg_key, "msg_value": cjcm_store[msg_key]}

@app.get("/cjcm/allmessages")
def retrieve_all_messages():
    if not cjcm_store:
        raise HTTPException(status_code=404, detail="No messages found")
    return {"messages": [{"msg_key": k, "msg_value": v} for k, v in cjcm_store.items()]}



ccai_store: Dict[str, str] = {}

# Input model
class Message(BaseModel):
    msg_key: str
    msg_value: str
    #read: boolean  # Default value for read status

@app.post("/ccai/store/")
def store_temp(messages: list[Message]):
    for message in messages:
            ccai_store[message.msg_key] = message.msg_value
    return {"message": f"Stored {len(messages)} messages "}

@app.get("/ccai/retrieve/{msg_key}")
def retrieve_temp(msg_key: str):
    #print("message1==>", ccai_store["message1"]);
    #print("message2==>", ccai_store["message2"]);

    num = random.randint(1, 1)  # random number between 1 and 3
    print(num)
    msg_key = msg_key+str(num)
    print(msg_key)
    if msg_key not in ccai_store:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"msg_key": msg_key, "msg_value": ccai_store[msg_key]}

@app.get("/ccai/allmessages")
def retrieve_all_messages():
    if not ccai_store:
        raise HTTPException(status_code=404, detail="No messages found")
    return {"messages": [{"msg_key": k, "msg_value": v} for k, v in ccai_store.items()]}




"""
http://127.0.0.1:8000/cjcm/store/

{
  "cjcm_msg_key": "message1",
  "cjcm_msg_value": "case details available, details are here ticket number123, customer facing issues with his new handset, related to network trobule shooting"
}

{
  "cjcm_msg_key": "message2",
  "cjcm_msg_value": "case details are not available"
}

http://127.0.0.1:8000/cjcm/retrieve/message1
http://127.0.0.1:8000/cjcm/retrieve/message2
"""
