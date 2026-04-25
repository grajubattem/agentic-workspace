# Interview Agent Skeletons

This repository provides four alternative starting points (skeletons) for building the coding exercise:

- Agents SDK (OpenAI Python SDK)
- LangChain
- LangGraph
- CrewAI

Each skeleton is intentionally minimal. No LLM or agent logic is implemented. You will wire up tools and reasoning yourself.

## Coding Exercise Instructions

### Objective
Build a chat-based agent that can answer resident questions and perform actions using JSON files as the data source.

### Data Files
- `residents.json`: Resident info, rent, balance, payments, ledger charges
- `maintenance_requests.json`: Maintenance service requests
- `property_info.json`: Office hours, amenities, community rules, upcoming events
- `event_signups.json`: Resident event registrations
- `packages.json`: Package availability for residents

### Requirements
- Accept natural language queries from residents (e.g., “What’s my balance?”)
- Use tools/functions to read from and write to the JSON files
- Support actions like:
  - Viewing rent, balance, property info, events, maintenance requests, packages

You may use any framework or language of your choice.

### Sample Resident Queries

#### Rent & Balance
- "What’s my open balance?"
- "How much is next month’s rent?"
- "Show me my rent payments for this year."

#### Property Info
- "What are the office hours on Saturday?"
- "List the amenities available."
- "What are the community rules?"

#### Events
- "What events are coming up?"
- "Show me events in November."
- "List my event signups."

#### Maintenance
- "List my maintenance requests."

#### Packages
- "Do I have any packages to collect?"
- "How many packages are waiting for me?"

## Setup

1. Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set OPENAI_API_KEY in .env if you plan to integrate a model later
4. Ensure JSON data files are present under `tools_data/` as described in the Coding Exercise Instructions section above.

## Choose a framework and run a skeleton

All skeletons provide a simple CLI that either runs once with `--prompt` or starts a REPL if `--prompt` is omitted.

- Agents SDK skeleton
  ```bash
  python -m frameworks.agents_sdk.main --prompt "What’s my balance?"
  ```

- LangChain skeleton
  ```bash
  python -m frameworks.langchain.main --prompt "List the amenities."
  ```

- LangGraph skeleton
  ```bash
  python -m frameworks.langgraph.main --prompt "Sign me up for event E003."
  ```

- CrewAI skeleton
  ```bash
  python -m frameworks.crewai.main --prompt "List my event signups."
  ```

Each will return a placeholder response until you implement the internals.

## Where to start coding

- Agents SDK: edit `frameworks/agents_sdk/agent.py` and `frameworks/agents_sdk/tools.py`
- LangChain: edit `frameworks/langchain/agent.py` and `frameworks/langchain/tools.py`
- LangGraph: edit `frameworks/langgraph/graph.py` and nodes under `frameworks/langgraph/nodes/`
- CrewAI: edit `frameworks/crewai/crew.py` and `frameworks/crewai/tools.py`

You can use the IO helpers in each framework’s `data_io.py` for reading/writing JSON under `tools_data/`.
