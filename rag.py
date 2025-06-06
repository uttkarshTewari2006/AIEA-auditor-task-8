import os
import getpass
from dotenv import load_dotenv

# Load .env and set API key before importing anything that uses it
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY is not set.")
    exit(1)

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from pyswip import Prolog
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ["OPENAI_API_KEY"])

raw_documents = [
    "Bob and Liz are Tom's parents.",
    "Ann and Pat are Bob's parents.",
    "Jim is Pat's parent",
    "Bill and Mary are Liz's parents",
    "Joe and Sue are Mary's parents",
    "Tom, Bob, Bill, Jim and Joe are males",
    "Liz, Ann, Pat, Mary and Sue are females",
]

docs_for_rag = [
    Document(page_content=text, metadata={"source": f"doc_{i}"})
    for i, text in enumerate(raw_documents)
]

# Use InMemoryVectorStore instead of FAISS
vector_store = InMemoryVectorStore.from_documents(docs_for_rag, embedding=embeddings)

kb = Prolog()
kb.consult("family_tree.kb")

@tool
def query_knowledge_base(query: str) -> str:
    """
    Queries the knowledge base with the provided query string.
    Returns formatted results as a string.
    """
    try:
        results = list(kb.query(query))
        if results:
            return f"Query results: {results}"
        else:
            return "No results found for the query."
    except Exception as e:
        return f"Error executing query: {str(e)}"

def get_context(question: str) -> list:
    """
    Performs semantic similarity search using the question
    Returns relevant documents as context
    """
    results = vector_store.similarity_search(question, k=3)
    return [doc.page_content for doc in results]

tools = [query_knowledge_base]

prompt = ChatPromptTemplate([
    ("system", "You are a knowledgeable assistant that can answer questions about a family tree which you have access to. "
    "You can also query a Prolog knowledge base for specific information."
    "Translate the question into a valid prolog query. To query if X is b's parent use the query 'parent(X, b)' where X is the parent of b. "
    "To query if b is X's sibling use the query 'sibling(b, X)'. Here are the facts: {facts}. The question will be"
    "provided to you in the user message."),
    ("user", "{question}")
])

def answer_question(question):
    facts = get_context(question)
    filled_prompt = prompt.format(
        question=question,
        facts=facts
    )
    prolog_query = llm.invoke(filled_prompt).content.strip()
    if not prolog_query.endswith("."):
        prolog_query += "."
    print(f"\n[DEBUG] Generated Prolog query:\n{prolog_query}\n")
    results = query_knowledge_base(prolog_query)
    if not results:
        return "No matching results."
    return results

if __name__ == "__main__":
    print("\n=== FAMILY TREE ASSISTANT ===")
    print("Type a family tree question or 'quit' to exit.\n")
    while True:
        user_input = input("Your question> ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        print("\nAnswer(s):")
        print(answer_question(user_input))
        print()