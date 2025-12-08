import streamlit as st
import os
from typing import Annotated, Literal, TypedDict
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- 1. CONFIG & FUTURISTIC UI ---
st.set_page_config(page_title="Nexus AI Agent", page_icon="🧬", layout="wide")

def inject_custom_css():
    st.markdown(
        """
        <style>
            /* IMPORT FONT */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
            
            /* GLOBAL THEME */
            .stApp {
                background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
                font-family: 'Inter', sans-serif;
                color: #e2e8f0;
            }
            
            /* HIDE DEFAULT STREAMLIT ELEMENTS */
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}

            /* GLASSMORPHISM CARDS */
            .glass-card {
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 24px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
                margin-bottom: 20px;
            }

            /* NEON HEADERS */
            .neon-text {
                background: linear-gradient(to right, #22d3ee, #a855f7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 700;
                font-size: 2.5rem;
            }
            
            /* CUSTOM INPUT FIELD */
            .stTextInput input {
                background-color: rgba(15, 23, 42, 0.8) !important;
                color: white !important;
                border: 1px solid #334155 !important;
                border-radius: 10px;
            }
            .stTextInput input:focus {
                border-color: #22d3ee !important;
                box-shadow: 0 0 10px rgba(34, 211, 238, 0.3) !important;
            }

            /* CUSTOM BUTTON */
            .stButton>button {
                background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 0.6rem 1.2rem !important;
                font-weight: 600 !important;
                transition: transform 0.2s, box-shadow 0.2s !important;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(124, 58, 237, 0.5) !important;
            }

            /* FOOTER */
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background: rgba(15, 23, 42, 0.9);
                backdrop-filter: blur(5px);
                color: #94a3b8;
                text-align: center;
                padding: 12px;
                border-top: 1px solid #1e293b;
                z-index: 100;
                font-size: 0.9rem;
            }
            .footer a {
                color: #38bdf8;
                text-decoration: none;
                margin: 0 10px;
                font-weight: 600;
            }
            .footer a:hover {
                color: #a855f7;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_css()

# --- 2. AGENT LOGIC ---
@tool
def web_search(query: str):
    """Search the web for information."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

tools = [web_search]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: AgentState):
    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    
    if not api_key:
        return {"messages": [("assistant", "⚠️ **System Alert:** API Key missing. Check Settings.")]}

    # CHANGED MODEL NAME TO FIX 404 ERROR
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        temperature=0,
        google_api_key=api_key
    ).bind_tools(tools)
    
    return {"messages": [llm.invoke(state["messages"])]}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    return "__end__"

@st.cache_resource
def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()

app = create_graph()

# --- 3. UI LAYOUT ---

# Title Section with Neon Effect
st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <h1 class="neon-text">NEXUS AGENT</h1>
        <p style="color: #94a3b8; margin-top: -10px;">Autonomous Research Intelligence System</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #e2e8f0; margin-bottom: 10px;">📡 Mission Control</h3>', unsafe_allow_html=True)
    topic = st.text_input("Research Target:", placeholder="e.g. Quantum Computing updates")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Initialize Search Protocol 🚀", use_container_width=True):
        if not topic:
            st.warning("⚠️ Target undefined.")
        else:
            st.session_state['run'] = True
            st.session_state['topic'] = topic
    st.markdown('</div>', unsafe_allow_html=True)

    # Info Box
    st.markdown("""
    <div class="glass-card" style="font-size: 0.85rem; color: #94a3b8;">
        <strong>System Status:</strong> <span style="color:#4ade80">ONLINE</span><br>
        <strong>Model:</strong> Gemini 1.5 Flash<br>
        <strong>Capabilities:</strong> Live Web Search
    </div>
    """, unsafe_allow_html=True)

with col2:
    if st.session_state.get('run'):
        # Status Output
        with st.status("🔄 **Processing Neural Query...**", expanded=True) as status:
            inputs = {"messages": [("user", f"Research: '{st.session_state['topic']}'. Write a structured report.")]}
            try:
                for event in app.stream(inputs):
                    for k, v in event.items():
                        if k == "agent":
                            msg = v["messages"][0]
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                query = msg.tool_calls[0]['args'].get('query')
                                st.markdown(f"**🌐 Scanning Network:** `{query}`")
                            else:
                                st.markdown("**⚡ Synthesizing Data...**")
                
                final = app.invoke(inputs)["messages"][-1].content
                status.update(label="✅ **Mission Accomplished**", state="complete", expanded=False)
                
                # Final Report Card
                st.markdown(f"""
                    <div class="glass-card">
                        <div style="display:flex; align-items:center; margin-bottom:15px; border-bottom:1px solid #334155; padding-bottom:10px;">
                            <span style="font-size:1.5rem; margin-right:10px;">📄</span>
                            <h2 style="margin:0; color:#f1f5f9;">Analysis Report</h2>
                        </div>
                        <div style="color: #cbd5e1; line-height: 1.6;">
                            {final}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"System Malfunction: {e}")
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 50px;">
            <div style="font-size: 3rem; margin-bottom: 20px;">🌌</div>
            <h3 style="color: #e2e8f0;">Awaiting Input</h3>
            <p style="color: #64748b;">Enter a research topic on the left to activate the agent.</p>
        </div>
        """, unsafe_allow_html=True)

# --- 4. FOOTER ---
st.markdown("""
    <div class="footer">
        Engineered by <span style="color: #a855f7;">R NITHYANANDACHARI</span> &nbsp;|&nbsp; 
        <a href="https://linkedin.com/in/Nithyananda" target="_blank">LinkedIn</a>
        <a href="https://github.com/Nithyaviswak" target="_blank">GitHub</a>
    </div>
""", unsafe_allow_html=True)