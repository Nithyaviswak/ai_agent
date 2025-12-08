import streamlit as st
import os
from typing import Annotated, Literal, TypedDict
from langchain_core.tools import tool
# from langchain_ollama import ChatOllama # <--- Commented out for Cloud
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

# Load environment variables (for local run)
load_dotenv()

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Agentic Research", page_icon="🕵️‍♂️", layout="wide")

def inject_tailwind():
    st.markdown(
        """
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .stApp { background-color: #f8fafc; }
            /* Footer fixed at bottom styling */
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: white;
                color: #4b5563;
                text-align: center;
                padding: 1rem;
                border-top: 1px solid #e5e7eb;
                z-index: 100;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_tailwind()

# --- 2. AGENT LOGIC (Gemini) ---
@tool
def web_search(query: str):
    """Search the web for information."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

tools = [web_search]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: AgentState):
    # Try getting key from Streamlit secrets (Cloud), otherwise os.environ (Local)
    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=api_key
    ).bind_tools(tools)
    
    return {"messages": [llm.invoke(state["messages"])]}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    if state["messages"][-1].tool_calls:
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
st.markdown("""
    <div class="bg-white p-6 rounded-lg shadow-sm mb-6 border-l-4 border-indigo-600">
        <h1 class="text-3xl font-bold text-gray-800">🕵️‍♂️ Autonomous Research Agent</h1>
        <p class="text-gray-500 mt-2">Powered by <strong>Gemini 1.5 Flash</strong></p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="bg-white p-6 rounded-lg shadow-sm h-full">', unsafe_allow_html=True)
    topic = st.text_input("Enter Topic:", placeholder="e.g. AI in Healthcare")
    
    if st.button("Start Research 🚀", use_container_width=True):
        if not topic:
            st.warning("Please enter a topic.")
        else:
            st.session_state['run'] = True
            st.session_state['topic'] = topic
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if st.session_state.get('run'):
        with st.status("🔄 **Agent Working...**", expanded=True) as status:
            inputs = {"messages": [("user", f"Research: '{st.session_state['topic']}'. Report with bullet points.")]}
            for event in app.stream(inputs):
                for k, v in event.items():
                    if k == "agent":
                        msg = v["messages"][0]
                        if msg.tool_calls:
                            st.write(f"🔎 Searching: `{msg.tool_calls[0]['args'].get('query')}`")
                        else:
                            st.write("📝 Writing Report...")
            status.update(label="Done!", state="complete", expanded=False)
        
        final = app.invoke(inputs)["messages"][-1].content
        st.markdown(f"""
            <div class="bg-white p-8 rounded-xl shadow-lg mt-4 prose max-w-none">
                {final}
            </div>
        """, unsafe_allow_html=True)

st.markdown("""
    <div class="footer">
        <div class="flex flex-col items-center justify-center">
            <p class="mb-2 text-sm font-semibold">Developed by <span class="text-indigo-600">R NITHYANANDACHARI</span></p>
            <div class="flex space-x-4">
                <a href="https://linkedin.com/in/Nithyananda1311" target="_blank" class="text-gray-500 hover:text-blue-600 transition">
                    <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                </a>
                <a href="https://github.com/Nithyaviswak" target="_blank" class="text-gray-500 hover:text-black transition">
                    <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                </a>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)