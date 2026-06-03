import os
os.environ["LANGCHAIN_DEFAULT_EMBEDDINGS"] = "False"
import streamlit as st
import uuid
import json
from utils.prepare_vectordb import get_vectorstore

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SUBTLE PROFESSIONAL STYLING ====================
st.markdown("""
    <style>
    /* Clean, professional color palette */
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --bg-hover: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --border: #475569;
        --accent: #ec4899;
    }
    
    /* Main container */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 100%) !important;
    }
    
    [data-testid="stSidebar"] {
        background: #1e293b !important;
        border-right: 1px solid #334155;
    }
    
    /* Text colors */
    body, p, span, div {
        color: #f1f5f9;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
        font-weight: 600;
    }
    
    /* Button styling - subtle */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #7c3aed 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 500 !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.2) !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Tab styling */
    .stTabs [role="tablist"] {
        background: transparent;
        border-bottom: 1px solid #334155;
        gap: 8px;
    }
    
    .stTabs [role="tab"] {
        color: #cbd5e1 !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 12px 16px !important;
        background: transparent !important;
        border-radius: 0 !important;
        transition: all 0.3s ease !important;
        font-weight: 500;
    }
    
    .stTabs [role="tab"]:hover {
        color: #6366f1 !important;
        border-bottom-color: rgba(99, 102, 241, 0.3) !important;
    }
    
    .stTabs [role="tab"][aria-selected="true"] {
        color: #6366f1 !important;
        border-bottom-color: #6366f1 !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stChatInputContainer input {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        padding: 12px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stChatInputContainer input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: #1e293b !important;
        border-left: 3px solid #334155 !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    
    /* Card styling */
    .card-subtle {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        transition: all 0.3s ease;
    }
    
    .card-subtle:hover {
        border-color: #475569;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Metric cards */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
        margin: 8px 0;
    }
    
    .metric-label {
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        background: rgba(99, 102, 241, 0.1);
        color: #6366f1;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Alert styling */
    .stAlert {
        background: rgba(99, 102, 241, 0.05) !important;
        border-left: 3px solid #6366f1 !important;
        border-radius: 8px !important;
    }
    
    /* Divider */
    hr {
        border-color: #334155 !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Sidebar button full width */
    .stSidebar .stButton > button {
        width: 100% !important;
        margin-bottom: 6px !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #7c3aed 100%) !important;
        border-radius: 4px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: transparent !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(99, 102, 241, 0.05) !important;
        border-color: #475569 !important;
    }
    
    /* Title section */
    .header-section {
        padding: 32px 0 24px 0;
        border-bottom: 1px solid #334155;
        margin-bottom: 24px;
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
    }
    
    .header-subtitle {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-top: 8px;
        font-weight: 400;
    }
    
    </style>
""", unsafe_allow_html=True)

class RefinedApp:
    def __init__(self):
        if not os.path.exists("docs"):
            os.makedirs("docs")
        if not os.path.exists("faiss_indexes"):
            os.makedirs("faiss_indexes")
        
        if "chats" not in st.session_state:
            st.session_state.chats = {}
        if "active_chat_id" not in st.session_state:
            st.session_state.active_chat_id = None
        
        self.recover_sessions()

    def save_messages(self, chat_id):
        """Save messages to JSON file"""
        msg_path = os.path.join("faiss_indexes", f"{chat_id}_messages.json")
        messages = st.session_state.chats[chat_id].get("messages", [])
        with open(msg_path, "w") as f:
            json.dump(messages, f, indent=2)

    def recover_sessions(self):
        """Recover sessions with their messages and FAISS indexes"""
        if not os.path.exists("docs"):
            return
        
        docs_files = os.listdir("docs")
        existing_file_sessions = {chat_data.get("file"): chat_id 
                                  for chat_id, chat_data in st.session_state.chats.items()}
        
        for file in docs_files:
            if file not in existing_file_sessions:
                session_id = str(uuid.uuid4())
                vectordb = None
                messages = []
                
                if os.path.exists("faiss_indexes"):
                    for idx_file in os.listdir("faiss_indexes"):
                        if idx_file.endswith("_metadata.json"):
                            try:
                                with open(os.path.join("faiss_indexes", idx_file), "r") as f:
                                    metadata = json.load(f)
                                    if metadata.get("file") == file:
                                        old_session_id = metadata.get("session_id")
                                        faiss_path = os.path.join("faiss_indexes", f"{old_session_id}.faiss")
                                        msg_path = os.path.join("faiss_indexes", f"{old_session_id}_messages.json")
                                        
                                        # Load messages
                                        if os.path.exists(msg_path):
                                            try:
                                                with open(msg_path, "r") as mf:
                                                    messages = json.load(mf)
                                            except:
                                                pass
                                        
                                        # Load FAISS index
                                        if os.path.exists(faiss_path):
                                            try:
                                                from langchain_community.vectorstores import FAISS
                                                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                                                embedding = GoogleGenerativeAIEmbeddings(
                                                    model="models/gemini-embedding-001",
                                                    google_api_key="AIzaSyABHiuNZOUyBzyvLzmEypTID0LFGAYIm0Y"
                                                )
                                                vectordb = FAISS.load_local(
                                                    faiss_path,
                                                    embedding,
                                                    allow_dangerous_deserialization=True
                                                )
                                            except:
                                                pass
                                        break
                            except:
                                pass
                
                st.session_state.chats[session_id] = {
                    "name": file,
                    "messages": messages,
                    "vector_db": vectordb,
                    "file": file,
                    "file_uploader_counter": 0
                }

    def render_sidebar(self):
        with st.sidebar:
            st.markdown("""
                <div style="padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 20px;">
                    <h2 style="margin: 0; font-size: 1.4rem;">Workspaces</h2>
                    <p style="color: #cbd5e1; font-size: 0.85rem; margin: 6px 0 0 0;">Manage sessions</p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("+ New Session", use_container_width=True, type="primary"):
                new_id = str(uuid.uuid4())
                st.session_state.chats[new_id] = {
                    "name": "New Session",
                    "messages": [],
                    "vector_db": None,
                    "file": None,
                    "file_uploader_counter": 0
                }
                st.session_state.active_chat_id = new_id
                st.rerun()
            
            st.divider()
            
            if not st.session_state.chats:
                st.info("No sessions yet", icon="ℹ️")
            else:
                for chat_id, chat_data in st.session_state.chats.items():
                    cols = st.columns([1, 0.15])
                    
                    with cols[0]:
                        label = chat_data["name"][:22]
                        indicator = "●" if chat_data.get("vector_db") else "○"
                        if st.button(f"{indicator} {label}", key=chat_id, use_container_width=True):
                            st.session_state.active_chat_id = chat_id
                            st.rerun()
                    
                    with cols[1]:
                        if st.button("×", key=f"del_{chat_id}", help="Delete"):
                            st.session_state["confirm_delete"] = chat_id

    def render_header(self):
        st.markdown("""
            <div class="header-section">
                <h1 class="header-title">Document Intelligence</h1>
                <p class="header-subtitle">Multi-session document analysis with advanced RAG capabilities</p>
            </div>
        """, unsafe_allow_html=True)

    def render_metrics(self):
        col1, col2, col3, col4 = st.columns(4)
        
        total_sessions = len(st.session_state.chats)
        total_files = len([c for c in st.session_state.chats.values() if c.get("file")])
        total_msgs = sum(len(c.get("messages", [])) for c in st.session_state.chats.values())
        active_sessions = len([c for c in st.session_state.chats.values() if c.get("vector_db")])
        
        with col1:
            st.markdown(f"""
                <div class="card-subtle" style="text-align: center;">
                    <div class="metric-label">Sessions</div>
                    <div class="metric-value">{total_sessions}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="card-subtle" style="text-align: center;">
                    <div class="metric-label">Documents</div>
                    <div class="metric-value">{total_files}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="card-subtle" style="text-align: center;">
                    <div class="metric-label">Messages</div>
                    <div class="metric-value">{total_msgs}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="card-subtle" style="text-align: center;">
                    <div class="metric-label">Ready</div>
                    <div class="metric-value">{active_sessions}</div>
                </div>
            """, unsafe_allow_html=True)

    def run(self):
        self.render_sidebar()
        self.render_header()
        
        # Delete confirmation
        if "confirm_delete" in st.session_state:
            chat_id = st.session_state["confirm_delete"]
            with st.container():
                st.warning(f"Delete '{st.session_state.chats[chat_id]['name']}'?")
                c1, c2 = st.columns(2)
                
                if c1.button("Confirm Delete", key="confirm_yes"):
                    file_to_remove = st.session_state.chats[chat_id].get("file")
                    if file_to_remove:
                        file_path = os.path.join("docs", file_to_remove)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    
                    faiss_path = os.path.join("faiss_indexes", f"{chat_id}.faiss")
                    if os.path.exists(faiss_path):
                        os.remove(faiss_path)
                    
                    metadata_path = os.path.join("faiss_indexes", f"{chat_id}_metadata.json")
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                    
                    msg_path = os.path.join("faiss_indexes", f"{chat_id}_messages.json")
                    if os.path.exists(msg_path):
                        os.remove(msg_path)
                    
                    del st.session_state.chats[chat_id]
                    if st.session_state.active_chat_id == chat_id:
                        st.session_state.active_chat_id = None
                    del st.session_state["confirm_delete"]
                    st.rerun()
                
                if c2.button("Cancel", key="confirm_no"):
                    del st.session_state["confirm_delete"]
                    st.rerun()
        
        self.render_metrics()
        st.divider()
        
        if st.session_state.active_chat_id:
            current_chat = st.session_state.chats[st.session_state.active_chat_id]
            
            tab1, tab2, tab3 = st.tabs(["Chat", "Documents", "Settings"])
            
            # TAB 1: CHAT
            with tab1:
                col_main, col_side = st.columns([0.7, 0.3])
                
                with col_main:
                    st.markdown(f"#### {current_chat.get('name', 'Untitled')}")
                    
                    if current_chat.get("vector_db"):
                        st.markdown(f"<div class='status-badge'>Ready</div>", unsafe_allow_html=True)
                        
                        for msg in current_chat.get("messages", []):
                            with st.chat_message(msg["role"]):
                                st.markdown(msg["content"])
                        
                        if user_query := st.chat_input("Type your question..."):
                            with st.chat_message("user"):
                                st.markdown(user_query)
                            current_chat["messages"].append({"role": "user", "content": user_query})
                            self.save_messages(st.session_state.active_chat_id)
                            
                            from utils.chatbot import get_response
                            with st.spinner("Analyzing..."):
                                response, ctx = get_response(user_query, [], current_chat["vector_db"])
                            
                            with st.chat_message("assistant"):
                                st.markdown(response)
                            current_chat["messages"].append({"role": "assistant", "content": response})
                            self.save_messages(st.session_state.active_chat_id)
                    else:
                        st.info("Upload a document to start chatting")
                
                with col_side:
                    st.markdown("#### Session Info")
                    st.markdown(f"""
                        <div class="card-subtle">
                            <div style="margin: 12px 0;">
                                <div class="metric-label">File</div>
                                <div style="color: #f1f5f9; font-weight: 500; margin-top: 4px;">{current_chat.get('file', 'None')}</div>
                            </div>
                            <hr style="margin: 12px 0;">
                            <div style="margin: 12px 0;">
                                <div class="metric-label">Messages</div>
                                <div style="color: #f1f5f9; font-weight: 500; margin-top: 4px;">{len(current_chat.get('messages', []))}</div>
                            </div>
                            <hr style="margin: 12px 0;">
                            <div style="margin: 12px 0;">
                                <div class="metric-label">Status</div>
                                <div style="color: #6366f1; font-weight: 500; margin-top: 4px;">{'Active' if current_chat.get('vector_db') else 'Pending'}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # TAB 2: DOCUMENTS
            with tab2:
                if not current_chat.get("vector_db"):
                    with st.expander("Upload Document", expanded=True):
                        unique_key = f"{st.session_state.active_chat_id}_{current_chat['file_uploader_counter']}"
                        uploaded_file = st.file_uploader(
                            "Select file (PDF, DOCX, TXT, CSV, PNG, JPG)",
                            type=["pdf", "docx", "txt", "csv", "png", "jpg", "jpeg"],
                            key=unique_key
                        )
                        
                        if uploaded_file:
                            progress = st.progress(0)
                            
                            file_path = os.path.join("docs", uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            progress.progress(25)
                            
                            vectordb = get_vectorstore([uploaded_file.name], from_session_state=False, session_id=st.session_state.active_chat_id)
                            current_chat["vector_db"] = vectordb
                            current_chat["name"] = uploaded_file.name
                            current_chat["file"] = uploaded_file.name
                            progress.progress(75)
                            
                            faiss_path = os.path.join("faiss_indexes", f"{st.session_state.active_chat_id}.faiss")
                            vectordb.save_local(faiss_path)
                            
                            metadata = {"file": uploaded_file.name, "session_id": st.session_state.active_chat_id}
                            with open(os.path.join("faiss_indexes", f"{st.session_state.active_chat_id}_metadata.json"), "w") as f:
                                json.dump(metadata, f)
                            
                            progress.progress(100)
                            st.success(f"Processed: {uploaded_file.name}")
                            st.rerun()
                else:
                    st.success(f"Active: {current_chat['name']}")
                
                st.divider()
                st.markdown("#### All Files")
                for chat_id, chat_data in st.session_state.chats.items():
                    if chat_data.get("file"):
                        col1, col2, col3 = st.columns([0.5, 0.3, 0.2])
                        with col1:
                            st.write(f"**{chat_data['file'][:35]}**")
                        with col2:
                            st.write(f"{len(chat_data.get('messages', []))} msgs")
                        with col3:
                            if st.button("Open", key=f"open_{chat_id}", use_container_width=True):
                                st.session_state.active_chat_id = chat_id
                                st.rerun()
            
            # TAB 3: SETTINGS
            with tab3:
                st.markdown("#### Configuration")
                
                if st.button("Reload Vectors", use_container_width=True):
                    if current_chat.get("file"):
                        try:
                            from langchain_community.vectorstores import FAISS
                            from langchain_google_genai import GoogleGenerativeAIEmbeddings
                            embedding = GoogleGenerativeAIEmbeddings(
                                model="models/gemini-embedding-001",
                                google_api_key="AIzaSyABHiuNZOUyBzyvLzmEypTID0LFGAYIm0Y"
                            )
                            faiss_path = os.path.join("faiss_indexes", f"{st.session_state.active_chat_id}.faiss")
                            vectordb = FAISS.load_local(
                                faiss_path,
                                embedding,
                                allow_dangerous_deserialization=True
                            )
                            current_chat["vector_db"] = vectordb
                            st.success("Vectors reloaded")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                if st.button("Clear Chat History", use_container_width=True):
                    current_chat["messages"] = []
                    self.save_messages(st.session_state.active_chat_id)
                    st.success("Chat cleared")
                    st.rerun()
        
        else:
            st.markdown("""
                <div style="text-align: center; padding: 80px 20px;">
                    <h2 style="margin: 0; color: #cbd5e1;">Start by creating a new session</h2>
                    <p style="color: #94a3b8; margin-top: 12px; font-size: 0.95rem;">
                    Use the sidebar to get started
                    </p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    app = RefinedApp()
    app.run()
