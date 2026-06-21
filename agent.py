from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# creates model
model = ChatOllama(model="llama3.2:3b")

# send question
#response = model.invoke("What is the capital of North Macedonia?")

#print(response.content)

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression like '2 + 2' or '15 * 8'. Use this whenever
    the user asks for a calculation, instead of working it out yourself."""
    try:
        return str(eval(expression))
    except Exception:
        return f"'{expression}' is not a valid math expression."



# give the agent a tool
agent = create_agent(
    model,
    tools=[calculator],
    checkpointer=InMemorySaver(),
    system_prompt="Answer questions directly and concisely. Only use the calculator tool for arithmetic.",
)

# ask question 
#result = agent.invoke(
#    {"messages": [{"role": "user", "content": "What is 47 * 89?"}]}
#)

# print answer
#print(result["messages"][-1].content)

#for m in result["messages"]:
#    print(m)

config = {"configurable": {"thread_id": "demo-1"}}

result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi my name is Katerina"}]},
    config=config,
)
print(result1["messages"][-1].content)

result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config=config,
)
print(result2["messages"][-1].content)