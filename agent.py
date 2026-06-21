from langchain_ollama import ChatOllama

# creates model
model = ChatOllama(model="llama3.2:3b")

# send question
response = model.invoke("What is the capital of North Macedonia?")

print(response.content)