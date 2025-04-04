import os
import operator
import json
from typing import TypedDict, Annotated, List

import streamlit as st
from dotenv import load_dotenv
import numexpr
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from phoenix.otel import register

# Load environment variables
load_dotenv()
print("OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY") is not None)

# OpenInference tracing (optional)
tracer_provider = register(project_name="langgraph", auto_instrument=True)

# Define tools
@tool
def calculator(expression: str) -> str:
    """
    Evaluates a simple mathematical expression like '2 * (3 + 5)'.
    Returns the result as a string.
    """
    result = numexpr.evaluate(expression).item()
    return str(result)

@tool
def save_to_pdf(content: str, filename: str = "output.pdf") -> str:
    """
    Saves the given text content to a PDF file.
    Returns a confirmation message or an error message.
    """
    PDF_OUTPUT_DIR = "."
    try:
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        filename = os.path.basename(filename)
        filepath = os.path.join(PDF_OUTPUT_DIR, filename)

        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph(content.replace('\n', '<br/>'), styles['Normal'])]
        doc.build(story)
        return f"Content successfully saved to {filepath}"
    except Exception as e:
        return f"Error saving content to PDF '{filename}': {e}"

# Tools
tools = [calculator, save_to_pdf]
tool_map = {tool.name: tool for tool in tools}

# Agent state

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# Initialize LLM
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print("LLM initialized and tools bound.")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm_with_tools = None

# Agent node
def agent_node(state: AgentState):
    if llm_with_tools is None:
        raise ValueError("LLM not initialized.")
    response = llm_with_tools.invoke(state['messages'])
    return {"messages": [response]}

# Tool node
def tool_node(state: AgentState):
    last_message = state['messages'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        if tool_name in tool_map:
            selected_tool = tool_map[tool_name]
            tool_input = tool_call['args']
            try:
                tool_output = selected_tool.invoke(tool_input)
            except Exception as e:
                tool_output = f"Error executing tool {tool_name}: {e}"
            tool_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call['id']))
        else:
            tool_messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call['id']))
    return {"messages": tool_messages}

# Router
def router(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if isinstance(last_message, ToolMessage):
        return "agent_node"
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        return "end"
    return "end"

# Workflow
def build_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("tool_node", tool_node)
    workflow.set_entry_point("agent_node")
    workflow.add_conditional_edges("agent_node", router, {"tool_node": "tool_node", "end": END})
    workflow.add_edge("tool_node", "agent_node")
    return workflow.compile()

# Compile app
app = build_workflow()
print("Graph compiled successfully!")

# Streamlit app
st.set_page_config(page_title="Agent Chat", page_icon="🤖")
st.title("Agent chat")

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar
st.sidebar.title("Settings")
st.sidebar.markdown("- Model: `gpt-4o-mini`")
st.sidebar.markdown("- Tools: Calculator, Save to PDF")

# Input box
user_input = st.chat_input("Type your message...")

# Process input
if user_input:
    st.session_state.conversation_history.append(HumanMessage(content=user_input))
    initial_state = {"messages": st.session_state.conversation_history}

    if app is None:
        st.error("App not compiled.")
    else:
        with st.spinner("Thinking..."):
            events = app.stream(initial_state, {"recursion_limit": 15})
            final_state_messages = []
            messages_from_this_run = []

            for event in events:
                step_name = list(event.keys())[0]
                if step_name != "__end__":
                    if "messages" in event[step_name]:
                        messages_from_this_run.extend(event[step_name]["messages"])
                if "__end__" in event:
                    final_state_data = event["__end__"]
                    final_state_messages = final_state_data.get("messages", [])
                    break

            st.session_state.conversation_history = (
                final_state_messages if final_state_messages else st.session_state.conversation_history + messages_from_this_run
            )

# Display chat
for msg in st.session_state.conversation_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            if msg.content:
                st.markdown(msg.content)
            elif msg.tool_calls:
                # If no content, but tool call exists, show tool call action
                tool_descriptions = []
                for tool_call in msg.tool_calls:
                    name = tool_call['name']
                    args = json.dumps(tool_call['args'])
                    tool_descriptions.append(f"Calling tool: `{name}` with args `{args}`")
                st.markdown("\n".join(tool_descriptions))
            else:
                st.markdown("(No content)")
    elif isinstance(msg, ToolMessage):
        with st.chat_message("tool"):
            st.markdown(f"**Tool Response:** {msg.content}")

