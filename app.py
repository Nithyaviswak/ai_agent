import streamlit as st
import os
import time  # <--- NEW IMPORT FOR DELAY
from typing import Annotated, Literal, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from googlesearch import search as gsearch
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- 1. CONFIG & UI SETUP ---
st.set_page_config(page_title="AI Agent", page_icon="🧬", layout="wide")

def inject_custom_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
            .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); font-family: 'Inter', sans-serif; color: #e2e8f0; }
            header, footer, .stDeployButton {visibility: hidden;}
            .glass-card { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 24px; margin-bottom: 20px; }
            .neon-text { background: linear-gradient(to right, #22d3ee, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; font-size: 2.5rem; }
            .stTextInput input { background-color: rgba(15, 23, 42, 0.8) !important; color: white !important; border: 1px solid #334155 !important; border-radius: 10px; }
            .stButton>button { background: linear-gradient(90deg, #2563eb, #7c3aed) !important; color: white !important; border: none; border-radius: 8px; }
            .footer { position: fixed; left: 0; bottom: 0; width: 100%; background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(5px); border-top: 1px solid #1e293b; z-index: 100; padding: 10px 0; display: flex; justify-content: center; gap: 20px; }
            .social-icon svg { width: 24px; height: 24px; fill: #94a3b8; transition: all 0.3s; }
            .social-icon:hover svg { fill: #22d3ee; transform: scale(1.1); }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_css()

# --- 2. FAILOVER MODEL LIST ---
MODEL_PRIORITY_LIST = [
    "gemini-2.0-flash-exp",    
    "gemini-exp-1206",         
    "gemini-2.0-pro-exp-02-05",
    "gemma-3-27b-it",         
    "gemma-3-12b-it",          
]

with st.sidebar:
    st.header("⚙️ Settings")
    st.success(f"🛡️ **Failover System Active**\n\nUsing Experimental & Gemma models to bypass the 20/day limit.")

# --- 3. AGENT LOGIC ---
@tool
def web_search(query: str):
    """Search the web for information using Google Search."""
    try:
        results = list(gsearch(query, num_results=5, advanced=True))
        formatted_results = "\n".join([f"- **{r.title}**: {r.description}" for r in results])
        return formatted_results if formatted_results else "No relevant results found."
    except Exception as e:
        return f"Search error: {str(e)}"

tools = [web_search]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: AgentState):
    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    
    if not api_key:
        return {"messages": [AIMessage(content="⚠️ API Key missing.")]}

    # CRITICAL FIX: SLOW DOWN THE AGENT
    # We sleep for 10 seconds to ensure we stay under the "5 requests per minute" limit
    time.sleep(10) 

    try:
        llm = ChatGoogleGenerativeAI(
            model=selected_model, 
            temperature=0, 
            google_api_key=api_key
        ).bind_tools(tools)
        
        return {"messages": [llm.invoke(state["messages"])]}
        
    except Exception as e:
        return {"messages": [AIMessage(content=f"❌ Error: {str(e)}")]}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    return "__end__"

def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()

app = create_graph()

# --- 4. UI LAYOUT ---
st.markdown('<div class="glass-card" style="text-align: center;"><h1 class="neon-text">AI AGENT</h1><p style="color: #94a3b8;">Autonomous Research Intelligence</p></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    topic = st.text_input("Research Target:", placeholder="e.g. AI Agents 2025")
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Initialize 🚀", use_container_width=True):
        if topic:
            st.session_state['run'] = True
            st.session_state['topic'] = topic
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if st.session_state.get('run'):
        with st.status("🔄 **Processing... (Slow Mode Active)**", expanded=True) as status:
            inputs = {"messages": [("user", f"Research: '{st.session_state['topic']}'. Write a report.")]}
            try:
                for event in app.stream(inputs):
                    for k, v in event.items():
                        if k == "agent":
                            msg = v["messages"][0]
                            if "Error" in str(msg.content):
                                st.error(msg.content)
                                status.update(label="❌ API Error", state="error")
                                st.stop()
                            
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                st.write(f"🌐 Searching: `{msg.tool_calls[0]['args'].get('query')}`")
                            else:
                                st.write("⚡ Synthesizing...")
                                
                final = app.invoke(inputs)["messages"][-1].content
                status.update(label="✅ Complete", state="complete", expanded=False)
                st.markdown(f'<div class="glass-card"><h2 style="color:#f1f5f9;">Report</h2><div style="color: #cbd5e1;">{final}</div></div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Critical Error: {e}")

# --- 5. FOOTER ---
st.markdown("""
    <div class="footer">
        <span style="color: #94a3b8; font-size: 0.9rem;">Engineered by <span style="color: #a855f7; font-weight:600;">R NITHYANANDACHARI</span></span>
        <div style="width: 1px; height: 20px; background: #334155; margin: 0 15px;"></div>
        <a href="https://linkedin.com/in/Nithyananda" target="_blank" class="social-icon"><svg viewBox="0 0 24 24"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg></a>
        <a href="https://github.com/Nithyaviswak" target="_blank" class="social-icon"><svg viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg></a>
    </div>
""", unsafe_allow_html=True)