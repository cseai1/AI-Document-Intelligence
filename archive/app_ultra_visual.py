import os
os.environ["LANGCHAIN_DEFAULT_EMBEDDINGS"] = "False"
import streamlit as st
import uuid
import json
from utils.prepare_vectordb import get_vectorstore

# ==================== ADVANCED STYLING ====================
st.set_page_config(
    page_title="AI Document Workspace",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced visual design
st.markdown("""
    <style>
    /* Root theme colors */
    :root {
        --primary: #00d96f;
        --primary-dark: #0fb366;
        --secondary: #1a1a2e;
        --tertiary: #16213e;
        --accent: #0f3460;
        --light: #eaeaea;
        --dark: #0a0e27;
    }
    
    /* Main container */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #0f3460 100%) !important;
        color: #eaeaea;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0a0e27 100%) !important;
        border-right: 2px solid rgba(0, 217, 111, 0.2);
    }
    
    /* Custom card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 217, 111, 0.1) 0%, rgba(15, 52, 96, 0.2) 100%);
        border: 1px solid rgba(0, 217, 111, 0.3);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 217, 111, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 217, 111, 0.6);
        box-shadow: 0 12px 40px rgba(0, 217, 111, 0.2);
        background: linear-gradient(135deg, rgba(0, 217, 111, 0.15) 0%, rgba(15, 52, 96, 0.3) 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d96f 0%, #0fb366 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 15px rgba(0, 217, 111, 0.3) !important;
        transition: all 0.3s ease !important;
        color: white !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.95em !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 217, 111, 0.5) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Tab styling */
    .stTabs [role="tablist"] {
        background: linear-gradient(90deg, rgba(22, 33, 62, 0.8) 0%, rgba(15, 52, 96, 0.8) 100%);
        border-bottom: 2px solid rgba(0, 217, 111, 0.2);
        border-radius: 12px;
        padding: 10px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [role="tab"] {
        background: transparent !important;
        color: rgba(234, 234, 234, 0.7) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border: 1px solid transparent !important;
    }
    
    .stTabs [role="tab"]:hover {
        color: #00d96f !important;
        border-color: rgba(0, 217, 111, 0.3) !important;
    }
    
    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 217, 111, 0.2) 0%, rgba(0, 217, 111, 0.05) 100%) !important;
        color: #00d96f !important;
        border-color: #00d96f !important;
        box-shadow: 0 0 20px rgba(0, 217, 111, 0.2) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(15, 52, 96, 0.4) 0%, rgba(0, 217, 111, 0.1) 100%) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(0, 217, 111, 0.2) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, rgba(15, 52, 96, 0.6) 0%, rgba(0, 217, 111, 0.15) 100%) !important;
        border-color: rgba(0, 217, 111, 0.4) !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: linear-gradient(135deg, rgba(22, 33, 62, 0.6) 0%, rgba(15, 52, 96, 0.4) 100%) !important;
        border-left: 4px solid rgba(0, 217, 111, 0.5) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stChatInputContainer input {
        background: linear-gradient(90deg, rgba(22, 33, 62, 0.8) 0%, rgba(15, 52, 96, 0.6) 100%) !important;
        border: 2px solid rgba(0, 217, 111, 0.2) !important;
        border-radius: 10px !important;
        color: #eaeaea !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stChatInputContainer input:focus {
        border-color: #00d96f !important;
        box-shadow: 0 0 20px rgba(0, 217, 111, 0.3) !important;
    }
    
    /* Metric styling */
    .stMetric {
        background: none !important;
    }
    
    .stMetricLabel {
        color: rgba(234, 234, 234, 0.8) !important;
    }
    
    .stMetricValue {
        color: #00d96f !important;
        font-size: 2rem !important;
    }
    
    /* Divider styling */
    hr {
        border-color: rgba(0, 217, 111, 0.2) !important;
        margin: 2rem 0 !important;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #eaeaea !important;
        font-weight: 700 !important;
    }
    
    /* Info boxes */
    .stAlert {
        background: linear-gradient(135deg, rgba(0, 217, 111, 0.1) 0%, rgba(15, 52, 96, 0.2) 100%) !important;
        border-left: 4px solid #00d96f !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d96f 0%, #0fb366 100%) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(0, 217, 111, 0.4) !important;
    }
    
    /* Sidebar button styling */
    .stSidebar .stButton > button {
        width: 100% !important;
        margin-bottom: 8px !important;
    }
    
    /* Loading spinner */
    .stSpinner > div > div {
        border-top-color: #00d96f !important;
    }
    
    /* Title section with glass morphism */
    .title-container {
        background: linear-gradient(135deg, rgba(0, 217, 111, 0.1) 0%, rgba(15, 52, 96, 0.2) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 217, 111, 0.3);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: slideDown 0.6s ease-out;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .title-main {
        color: #00d96f;
        font-size: 2.8em;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 0 20px rgba(0, 217, 111, 0.3);
        letter-spacing: -1px;
    }
    
    .title-sub {
        color: rgba(234, 234, 234, 0.9);
        font-size: 1.1em;
        margin-top: 12px;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* File upload area */
    .uploadedFile {
        background: linear-gradient(135deg, rgba(0, 217, 111, 0.15) 0%, rgba(15, 52, 96, 0.25) 100%) !important;
        border: 2px dashed rgba(0, 217, 111, 0.4) !important;
        border-radius: 15px !important;
        padding: 20px !important;
    }
    
    /* Animation for new elements */
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .animate-fade-in {
        animation: fadeInScale 0.4s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

class AdvancedVisualApp:
    def __init__(self):
        if not os.path.exists("docs"):
            os.makedirs("docs")
        
        if "chats" not in st.session_state:
            st.session_state.chats = {}
        if "active_chat_id" not in st.session_state:
            st.session_state.active_chat_id = None
        
        self.recover_sessions()

    def recover_sessions(self):
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
                                        if os.path.exists(faiss_path):
                                            try:
                                                from langchain_community.vectorstores import FAISS
                                                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                                                embedding = GoogleGenerativeAIEmbeddings(
                                                    model="models/gemini-embedding-001",
                                                    google_api_key="API KEY"
                                                )
                                                vectordb = FAISS.load_local(
                                                    faiss_path,
                                                    embedding,
                                                    allow_dangerous_deserialization=True
                                                )
                                            except:
                                                pass
                                        msg_path = os.path.join("faiss_indexes", f"{old_session_id}_messages.json")
                                        if os.path.exists(msg_path):
                                            try:
                                                with open(msg_path, "r") as mf:
                                                    messages = json.load(mf)
                                            except:
                                                messages = []
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

    def save_messages(self, chat_id):
        msg_path = os.path.join("faiss_indexes", f"{chat_id}_messages.json")
        try:
            with open(msg_path, "w") as mf:
                json.dump(st.session_state.chats[chat_id].get("messages", []), mf, indent=2)
        except:
            pass

    def render_sidebar(self):
        with st.sidebar:
            st.markdown("---")
            st.markdown("""
                <div style="text-align: center; padding: 20px 0;">
                    <h2 style="color: #00d96f; font-size: 1.8em; margin: 0;">💼 WORKSPACE</h2>
                    <p style="color: rgba(234, 234, 234, 0.6); font-size: 0.9em; margin-top: 5px;">Session Manager</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            
            if st.button("➕ NEW SESSION", use_container_width=True, type="primary"):
                new_id = str(uuid.uuid4())
                st.session_state.chats[new_id] = {
                    "name": "New Chat",
                    "messages": [],
                    "vector_db": None,
                    "file": None,
                    "file_uploader_counter": 0
                }
                st.session_state.active_chat_id = new_id
                st.rerun()
            
            st.markdown("---")
            st.markdown("""
                <p style="color: #00d96f; font-weight: 600; margin: 10px 0; font-size: 0.9em;">
                📋 CONVERSATIONS
                </p>
            """, unsafe_allow_html=True)
            
            if not st.session_state.chats:
                st.info("No sessions yet. Create one to start!", icon="ℹ️")
            else:
                for chat_id, chat_data in st.session_state.chats.items():
                    cols = st.columns([0.8, 0.2])
                    label = chat_data["name"][:18] + ("..." if len(chat_data["name"]) > 18 else "")
                    status = "🟢" if chat_data.get("vector_db") else "🔴"
                    
                    with cols[0]:
                        if st.button(f"{status} {label}", key=chat_id, use_container_width=True):
                            st.session_state.active_chat_id = chat_id
                            st.rerun()
                    
                    with cols[1]:
                        if st.button("🗑️", key=f"del_{chat_id}", help="Delete session"):
                            st.session_state["confirm_delete"] = chat_id

    def render_header(self):
        st.markdown("""
            <div class="title-container">
                <h1 class="title-main">🤖 Enterprise RAG Assistant</h1>
                <p class="title-sub">Multi-Modal Document Intelligence • Real-Time Analytics • Advanced Session Management</p>
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
                <div class="metric-card">
                    <div style="text-align: center;">
                        <div style="font-size: 2em; margin-bottom: 10px;">📊</div>
                        <div style="color: #00d96f; font-size: 2em; font-weight: 700;">{total_sessions}</div>
                        <div style="color: rgba(234, 234, 234, 0.7); font-size: 0.9em; margin-top: 5px;">Active Sessions</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="text-align: center;">
                        <div style="font-size: 2em; margin-bottom: 10px;">📄</div>
                        <div style="color: #00d96f; font-size: 2em; font-weight: 700;">{total_files}</div>
                        <div style="color: rgba(234, 234, 234, 0.7); font-size: 0.9em; margin-top: 5px;">Documents</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="text-align: center;">
                        <div style="font-size: 2em; margin-bottom: 10px;">💬</div>
                        <div style="color: #00d96f; font-size: 2em; font-weight: 700;">{total_msgs}</div>
                        <div style="color: rgba(234, 234, 234, 0.7); font-size: 0.9em; margin-top: 5px;">Interactions</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="text-align: center;">
                        <div style="font-size: 2em; margin-bottom: 10px;">⚡</div>
                        <div style="color: #00d96f; font-size: 2em; font-weight: 700;">{active_sessions}</div>
                        <div style="color: rgba(234, 234, 234, 0.7); font-size: 0.9em; margin-top: 5px;">Ready</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    def run(self):
        self.render_sidebar()
        self.render_header()
        
        # Delete confirmation
        if "confirm_delete" in st.session_state:
            chat_id = st.session_state["confirm_delete"]
            with st.container():
                st.warning(f"⚠️ Delete '{st.session_state.chats[chat_id]['name']}'?")
                c1, c2, c3 = st.columns([1, 1, 2])
                
                if c1.button("✓ Delete", key="confirm_yes"):
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
                
                if c2.button("✗ Cancel", key="confirm_no"):
                    del st.session_state["confirm_delete"]
                    st.rerun()
        
        self.render_metrics()
        st.markdown("---")
        
        if st.session_state.active_chat_id:
            current_chat = st.session_state.chats[st.session_state.active_chat_id]
            
            tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📁 Files", "📊 Analytics", "⚙️ Settings"])
            
            # TAB 1: CHAT
            with tab1:
                col_left, col_right = st.columns([0.65, 0.35])
                
                with col_left:
                    st.markdown(f"### 📄 {current_chat.get('name', 'Select File')}")
                    
                    if current_chat.get("vector_db"):
                        st.success("✓ Vector Index Ready", icon="✅")
                        
                        for msg in current_chat["messages"]:
                            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                                st.markdown(msg["content"])
                        
                        if user_query := st.chat_input("Ask something about your document...", key="chat_input"):
                            with st.chat_message("user", avatar="👤"):
                                st.markdown(user_query)
                            current_chat["messages"].append({"role": "user", "content": user_query})
                            
                            from utils.chatbot import get_response
                            with st.spinner("🔍 Analyzing document..."):
                                response, ctx = get_response(user_query, [], current_chat["vector_db"])
                            
                            with st.chat_message("assistant", avatar="🤖"):
                                st.markdown(response)
                            current_chat["messages"].append({"role": "assistant", "content": response})
                            self.save_messages(st.session_state.active_chat_id)
                    else:
                        st.info("⬆️ Upload a document to start chatting", icon="ℹ️")
                
                with col_right:
                    st.markdown("### 📋 Session Details")
                    with st.container():
                        st.markdown(f"""
                            <div class="metric-card">
                                <div style="margin: 10px 0;">
                                    <p style="color: rgba(234, 234, 234, 0.7); margin: 0; font-size: 0.9em;">📄 File</p>
                                    <p style="color: #00d96f; margin: 5px 0; font-weight: 600;">{current_chat.get('file', 'None')}</p>
                                </div>
                                <hr style="margin: 15px 0; border-color: rgba(0, 217, 111, 0.2);">
                                <div style="margin: 10px 0;">
                                    <p style="color: rgba(234, 234, 234, 0.7); margin: 0; font-size: 0.9em;">💬 Messages</p>
                                    <p style="color: #00d96f; margin: 5px 0; font-weight: 600;">{len(current_chat.get('messages', []))}</p>
                                </div>
                                <hr style="margin: 15px 0; border-color: rgba(0, 217, 111, 0.2);">
                                <div style="margin: 10px 0;">
                                    <p style="color: rgba(234, 234, 234, 0.7); margin: 0; font-size: 0.9em;">🟢 Status</p>
                                    <p style="color: #00d96f; margin: 5px 0; font-weight: 600;">{'Active' if current_chat.get('vector_db') else 'Pending'}</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
            
            # TAB 2: FILES
            with tab2:
                st.markdown("### 📂 Document Management")
                
                if not current_chat.get("vector_db"):
                    with st.expander("📤 Upload New Document", expanded=True):
                        if "file_uploader_counter" not in current_chat:
                            current_chat["file_uploader_counter"] = 0
                        
                        unique_key = f"{st.session_state.active_chat_id}_{current_chat['file_uploader_counter']}"
                        uploaded_file = st.file_uploader(
                            "Choose file (PDF, DOCX, TXT, CSV, PNG, JPG)",
                            type=["pdf", "docx", "txt", "csv", "png", "jpg", "jpeg"],
                            key=unique_key
                        )
                        
                        if uploaded_file:
                            progress = st.progress(0)
                            status = st.empty()
                            
                            status.text("📤 Uploading...")
                            progress.progress(20)
                            
                            file_path = os.path.join("docs", uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            status.text("⚙️ Processing...")
                            progress.progress(50)
                            
                            vectordb = get_vectorstore([uploaded_file.name], from_session_state=False, session_id=st.session_state.active_chat_id)
                            current_chat["vector_db"] = vectordb
                            current_chat["name"] = uploaded_file.name
                            current_chat["file"] = uploaded_file.name
                            
                            status.text("💾 Saving...")
                            progress.progress(80)
                            
                            if not os.path.exists("faiss_indexes"):
                                os.makedirs("faiss_indexes")
                            
                            faiss_path = os.path.join("faiss_indexes", f"{st.session_state.active_chat_id}.faiss")
                            vectordb.save_local(faiss_path)
                            metadata = {"file": uploaded_file.name, "session_id": st.session_state.active_chat_id}
                            with open(os.path.join("faiss_indexes", f"{st.session_state.active_chat_id}_metadata.json"), "w") as f:
                                json.dump(metadata, f)
                            self.save_messages(st.session_state.active_chat_id)
                            
                            progress.progress(100)
                            status.text("✅ Complete!")
                            st.success(f"✅ Successfully processed: {uploaded_file.name}")
                            st.balloons()
                            st.rerun()
                else:
                    st.success(f"✓ {current_chat['name']} is ready")
                
                st.divider()
                st.markdown("### 📚 All Files")
                for chat_id, chat_data in st.session_state.chats.items():
                    if chat_data.get("file"):
                        col1, col2, col3 = st.columns([0.5, 0.3, 0.2])
                        with col1:
                            st.write(f"📄 **{chat_data['file'][:30]}**")
                        with col2:
                            st.write(f"💬 {len(chat_data.get('messages', []))} messages")
                        with col3:
                            if st.button("View", key=f"view_{chat_id}", use_container_width=True):
                                st.session_state.active_chat_id = chat_id
                                st.rerun()
            
            # TAB 3: ANALYTICS
            with tab3:
                st.markdown("### 📊 Session Analytics")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                        <div class="metric-card">
                            <div style="text-align: center;">
                                <div style="font-size: 3em; margin-bottom: 10px;">📊</div>
                                <div style="color: #00d96f; font-size: 2em; font-weight: 700; margin: 10px 0;">""" + str(len(st.session_state.chats)) + """</div>
                                <div style="color: rgba(234, 234, 234, 0.7);">Total Sessions</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div class="metric-card">
                            <div style="text-align: center;">
                                <div style="font-size: 3em; margin-bottom: 10px;">📄</div>
                                <div style="color: #00d96f; font-size: 2em; font-weight: 700; margin: 10px 0;">""" + str(len([c for c in st.session_state.chats.values() if c.get("file")])) + """</div>
                                <div style="color: rgba(234, 234, 234, 0.7);">Documents</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                st.markdown("**Activity Breakdown**")
                for name, chat_data in [(c["name"], c) for c in st.session_state.chats.values()]:
                    progress_val = min(100, len(chat_data.get("messages", [])) * 10)
                    st.write(f"**{name[:35]}** • {len(chat_data.get('messages', []))} interactions")
                    st.progress(progress_val / 100)
            
            # TAB 4: SETTINGS
            with tab4:
                st.markdown("### ⚙️ Session Settings")
                
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="margin: 15px 0;">
                            <p style="color: rgba(234, 234, 234, 0.7); margin: 0; font-size: 0.85em;">SESSION ID</p>
                            <code style="color: #00d96f; font-size: 0.9em; word-break: break-all;">{st.session_state.active_chat_id[:16]}...</code>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                
                if st.button("🔄 Reload Vectors", use_container_width=True):
                    if current_chat.get("file"):
                        faiss_path = os.path.join("faiss_indexes", f"{st.session_state.active_chat_id}.faiss")
                        if os.path.exists(faiss_path):
                            from langchain_community.vectorstores import FAISS
                            from langchain_google_genai import GoogleGenerativeAIEmbeddings
                            embedding = GoogleGenerativeAIEmbeddings(
                                model="models/gemini-embedding-001",
                                google_api_key="API KEY"
                            )
                            vectordb = FAISS.load_local(
                                faiss_path,
                                embedding,
                                allow_dangerous_deserialization=True
                            )
                            current_chat["vector_db"] = vectordb
                            st.success("✓ Vectors reloaded!")
                
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    current_chat["messages"] = []
                    self.save_messages(st.session_state.active_chat_id)
                    st.success("✓ Chat cleared!")
                    st.rerun()
        
        else:
            st.markdown("""
                <div style="text-align: center; padding: 60px 20px;">
                    <div style="font-size: 3em; margin-bottom: 20px;">👈</div>
                    <h2 style="color: #00d96f; margin: 0;">Ready to Get Started?</h2>
                    <p style="color: rgba(234, 234, 234, 0.7); margin-top: 10px; font-size: 1.1em;">
                    Click <b>"➕ NEW SESSION"</b> in the sidebar to begin
                    </p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    app = AdvancedVisualApp()
    app.run()
