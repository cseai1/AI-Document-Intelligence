import os
os.environ["LANGCHAIN_DEFAULT_EMBEDDINGS"] = "False"
import streamlit as st
import uuid
import json
from utils.prepare_vectordb import get_vectorstore

class AdvancedChatApp:
    def __init__(self):
        if not os.path.exists("docs"):
            os.makedirs("docs")
        
        st.set_page_config(
            page_title="AI Document Workspace", 
            page_icon="🤖", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom theme
        st.markdown("""
            <style>
            :root { --primary-color: #10a37f; }
            [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%); }
            </style>
        """, unsafe_allow_html=True)
        
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
                                                    google_api_key="AIzaSyABHiuNZOUyBzyvLzmEypTID0LFGAYIm0Y"
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

    def run(self):
        # ==================== SIDEBAR ====================
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=48)
            st.title("💼 Workspace")
            
            # New chat button
            if st.button("➕ New Chat Session", use_container_width=True, type="primary"):
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
            
            st.write("---")
            st.caption("📋 Recent Conversations")
            
            to_delete = None
            confirm_delete = None
            
            for chat_id, chat_data in st.session_state.chats.items():
                cols = st.columns([0.8, 0.2])
                label = chat_data["name"][:20] + ("..." if len(chat_data["name"]) > 20 else "")
                
                with cols[0]:
                    if st.button(f"💬 {label}", key=chat_id, use_container_width=True):
                        st.session_state.active_chat_id = chat_id
                        st.rerun()
                
                with cols[1]:
                    if st.button("🗑️", key=f"delete_{chat_id}", use_container_width=True):
                        confirm_delete = chat_id
            
            # Delete confirmation
            if confirm_delete:
                st.session_state["show_confirm_delete"] = confirm_delete
            
            if "show_confirm_delete" in st.session_state:
                chat_id = st.session_state["show_confirm_delete"]
                st.warning(f"Delete '{st.session_state.chats[chat_id]['name']}'?")
                c1, c2 = st.columns(2)
                
                if c1.button("✓ Yes", key="yes_delete"):
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
                    del st.session_state["show_confirm_delete"]
                    st.rerun()
                
                if c2.button("✗ No", key="cancel_delete"):
                    del st.session_state["show_confirm_delete"]
                    st.rerun()
        
        # ==================== MAIN CONTENT ====================
        st.markdown("""
            <style>
            .title-section {
                background: linear-gradient(135deg, #10a37f 0%, #0f766e 100%);
                padding: 2rem;
                border-radius: 15px;
                margin-bottom: 2rem;
            }
            .title-main { color: white; font-size: 2.2em; font-weight: 700; margin: 0; }
            .title-sub { color: rgba(255,255,255,0.85); font-size: 1em; margin-top: 0.5rem; }
            </style>
            <div class="title-section">
                <h1 class="title-main">🤖 Enterprise RAG Assistant</h1>
                <p class="title-sub">Advanced Multi-Modal Document Analysis | Session Management | Real-Time Analytics</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.active_chat_id:
            current_chat = st.session_state.chats[st.session_state.active_chat_id]
            
            # ========== METRICS ROW ==========
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Active Sessions", len(st.session_state.chats), "📊")
            with col2:
                files_count = len([c for c in st.session_state.chats.values() if c.get("file")])
                st.metric("Documents", files_count, "📄")
            with col3:
                msgs_count = sum(len(c.get("messages", [])) for c in st.session_state.chats.values())
                st.metric("Interactions", msgs_count, "💬")
            with col4:
                status = "✓ Active" if current_chat.get("vector_db") else "⊗ Pending"
                st.metric("Session Status", status, "🔧")
            
            st.divider()
            
            # ========== TABS ==========
            tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📁 Files", "📊 Analytics", "⚙️ Settings"])
            
            # TAB 1: CHAT
            with tab1:
                col_left, col_right = st.columns([0.7, 0.3])
                
                with col_left:
                    st.subheader(f"📄 Document: {current_chat.get('name', 'Select File')}")
                    
                    # Chat display
                    if current_chat.get("vector_db"):
                        st.success("✓ Vector index loaded", icon="✅")
                        for msg in current_chat["messages"]:
                            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                                st.markdown(msg["content"])
                        
                        if user_query := st.chat_input("Ask something...", key="user_input"):
                            with st.chat_message("user", avatar="👤"):
                                st.markdown(user_query)
                            current_chat["messages"].append({"role": "user", "content": user_query})
                            
                            from utils.chatbot import get_response
                            with st.spinner("🔍 Analyzing..."):
                                response, ctx = get_response(user_query, [], current_chat["vector_db"])
                            
                            with st.chat_message("assistant", avatar="🤖"):
                                st.markdown(response)
                            current_chat["messages"].append({"role": "assistant", "content": response})
                            self.save_messages(st.session_state.active_chat_id)
                    else:
                        st.info("⬆️ Upload a document in the Files tab to start chatting")
                
                with col_right:
                    st.subheader("📋 Session Info")
                    st.write(f"**File:** {current_chat.get('file', 'None')}")
                    st.write(f"**Messages:** {len(current_chat.get('messages', []))}")
                    st.write(f"**Status:** {'🟢 Active' if current_chat.get('vector_db') else '🔴 Inactive'}")
            
            # TAB 2: FILE MANAGEMENT
            with tab2:
                st.subheader("📂 Manage Documents")
                
                if not current_chat.get("vector_db"):
                    with st.expander("📤 Upload New Document", expanded=True):
                        if "file_uploader_counter" not in current_chat:
                            current_chat["file_uploader_counter"] = 0
                        
                        unique_key = f"{st.session_state.active_chat_id}_{current_chat['file_uploader_counter']}"
                        uploaded_file = st.file_uploader(
                            "Choose file",
                            type=["pdf", "docx", "txt", "csv", "png", "jpg", "jpeg"],
                            key=unique_key
                        )
                        
                        if uploaded_file:
                            progress = st.progress(0)
                            
                            progress.progress(20)
                            file_path = os.path.join("docs", uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            progress.progress(50)
                            vectordb = get_vectorstore([uploaded_file.name], from_session_state=False, session_id=st.session_state.active_chat_id)
                            current_chat["vector_db"] = vectordb
                            current_chat["name"] = uploaded_file.name
                            current_chat["file"] = uploaded_file.name
                            
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
                            current_chat["file_uploader_counter"] += 1
                            st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                            st.balloons()
                            st.rerun()
                else:
                    st.info(f"✓ Document loaded: {current_chat['name']}" )
                
                st.divider()
                st.subheader("📚 All Uploaded Files")
                for chat_id, chat_data in st.session_state.chats.items():
                    if chat_data.get("file"):
                        col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
                        with col1:
                            st.write(f"📄 {chat_data['file']}")
                        with col2:
                            st.write(f"💬 {len(chat_data.get('messages', []))} msgs")
                        with col3:
                            if st.button("View", key=f"view_{chat_id}", use_container_width=True):
                                st.session_state.active_chat_id = chat_id
                                st.rerun()
            
            # TAB 3: ANALYTICS
            with tab3:
                st.subheader("📊 Analytics Dashboard")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Sessions", len(st.session_state.chats))
                    st.metric("Total Documents", len([c for c in st.session_state.chats.values() if c.get("file")]))
                
                with col2:
                    total_msgs = sum(len(c.get("messages", [])) for c in st.session_state.chats.values())
                    st.metric("Total Interactions", total_msgs)
                    active_with_vectors = len([c for c in st.session_state.chats.values() if c.get("vector_db")])
                    st.metric("Ready Sessions", active_with_vectors)
                
                st.divider()
                st.write("**Session Activity:**")
                for name, chat_data in [(c["name"], c) for c in st.session_state.chats.values()]:
                    progress_val = min(100, len(chat_data.get("messages", [])) * 10)
                    st.write(f"**{name[:30]}** - {len(chat_data.get('messages', []))} messages")
                    st.progress(progress_val / 100)
            
            # TAB 4: SETTINGS
            with tab4:
                st.subheader("⚙️ Session Settings")
                st.write(f"**Active Session ID:** `{st.session_state.active_chat_id[:8]}...`")
                st.write(f"**Document:** {current_chat.get('file', 'None')}")
                st.write(f"**Messages:** {len(current_chat.get('messages', []))}")
                
                if st.button("🔄 Reload Vectors", use_container_width=True):
                    if current_chat.get("file"):
                        faiss_path = os.path.join("faiss_indexes", f"{st.session_state.active_chat_id}.faiss")
                        if os.path.exists(faiss_path):
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
                            current_chat["vector_db"] = vectordb
                            st.success("✓ Vectors reloaded")
                
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    current_chat["messages"] = []
                    self.save_messages(st.session_state.active_chat_id)
                    st.success("✓ Chat history cleared")
                    st.rerun()
        
        else:
            st.info("👈 Click '➕ New Chat Session' to get started!", icon="ℹ️")

if __name__ == "__main__":
    app = AdvancedChatApp()
    app.run()
