from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# --- Model ---
model = ChatOllama(model="qwen2.5:3b")

# --- Tool 1: calculator ---
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression like '2 + 2' or '15 * 8'. Use this whenever
    the user asks for a calculation, instead of working it out yourself."""
    try:
        return str(eval(expression))
    except Exception:
        return f"'{expression}' is not a valid math expression."

# --- Tool 2: document retrieval ---
embedding = OllamaEmbeddings(model="nomic-embed-text")

docs = [
    "German visa expires on March 14, 2026",
    "Germany has 16 different federal states",
    "Oktoberfest is celebrated in Munich",
]

vectorstore = InMemoryVectorStore.from_texts(docs, embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

@tool
def search_docs(query: str) -> str:
    """Search the internal facts database. Use this whenever the user asks about
    a specific fact, project, place, or date."""
    results = retriever.invoke(query)
    return "\n".join(doc.page_content for doc in results)

# --- Agent: model + tools + memory ---
agent = create_agent(
    model,
    tools=[calculator, search_docs],
    checkpointer=InMemorySaver(),
    system_prompt="Answer questions directly and concisely. Use the calculator for arithmetic, and search_docs to look up facts in the database.",
)

# --- Demo ---
config = {"configurable": {"thread_id": "demo-1"}}

def ask(question):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    print(f"Q: {question}")
    print(f"A: {result['messages'][-1].content}\n")

ask("Hi, my name is Katerina.")           # stores info in memory
ask("What is 47 * 89?")                    # uses the calculator tool
ask("When does German visa expire?")   # uses the retrieval tool
ask("What's my name?")                     # recalls from memory