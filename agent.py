from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent

# creates model
model = ChatOllama(model="llama3.2:3b")

# send question
#response = model.invoke("What is the capital of North Macedonia?")

#print(response.content)

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression like '2 + 2' or '15 * 8'. Use this whenever
    the user asks for a calculation, instead of working it out yourself."""
    return str(eval(expression))



# give the agent a tool
agent = create_agent(model, tools=[calculator])

# ask question 
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is 47 * 89?"}]}
)

# print answer
print(result["messages"][-1].content)

for m in result["messages"]:
    print(m)