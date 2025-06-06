import langchain
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import getpass
import os
import regex
from pyswip import Prolog
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

kb = Prolog()
kb.consult("family_tree.kb")

@tool
def query_knowledge_base(query: str) -> str:
    """
    Queries the knowledge base with the provided query string.
    Returns formatted results as a string.
    
    Args:
        query: Prolog query string (e.g., "parent(X, bob)" to find bob's parents)
    """
    try:
        results = list(kb.query(query))
        if results:
            return f"Query results: {results}"
        else:
            return "No results found for the query."
    except Exception as e:
        return f"Error executing query: {str(e)}"

tools = [query_knowledge_base]

with open("family_tree.kb", "r") as file:
    kb_content = file.read()

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai").bind_tools(tools)

# Create proper system message with correct query examples
system_message = f"""You are an assistant that can query a family tree knowledge base using Prolog syntax.

IMPORTANT QUERY PATTERNS:
- To find someone's PARENTS: parent(X, person_name)
- To find someone's CHILDREN: parent(person_name, X)  
- To find siblings: sibling(X, person_name)

Knowledge base content:
{kb_content}

When you make a query, I will execute it and provide the results. Then provide a natural language answer based on the results.

When asked a question, respond with ONLY the Prolog query needed to answer it. Do NOT include explanations or code blocks. Example: male(tom)."""

# Initial conversation
messages = [
    SystemMessage(content=system_message),
    HumanMessage(content="Who are the parents of Bob?")
]

# Get the model's response (which should include a tool call)
response = model.invoke(messages)

print("Initial Response:")
print(f"Content: '{response.content}'")
print(f"Tool calls: {response.tool_calls}")

# Add the AI response to messages
messages.append(response)

# Execute tool calls and add tool messages
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"\nExecuting tool: {tool_call['name']}")
        print(f"Query: {tool_call['args']['query']}")
        
        # Execute the tool
        tool_result = query_knowledge_base(tool_call['args']['query'])
        print(f"Tool result: {tool_result}")
        
        # Add tool message
        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=tool_call['id']
        )
        messages.append(tool_message)

    # Get final response from the model
    final_response = model.invoke(messages)
    print(f"\nFinal Response:")
    print(f"Content: {final_response.content}")
else:
    print("No tool calls made")

print("\n" + "="*50)
print("ALTERNATIVE: Using AgentExecutor (Recommended)")

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Create a proper prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create agent
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Execute the query
try:
    result = agent_executor.invoke({"input": "Who are the parents of Bob?"})
    print("\nAgent result:")
    print(result["output"])
except Exception as e:
    print(f"Agent execution error: {e}")