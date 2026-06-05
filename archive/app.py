import os
os.environ["LANGCHAIN_DEFAULT_EMBEDDINGS"] = "False"
import streamlit as st
import os
import uuid
import json
from utils.save_docs import save_docs_to_vectordb
from utils.session_state import initialize_session_state_variables
from utils.prepare_vectordb import get_vectorstore
from utils.chatbot import chat

class ChatApp:
    def __init__(self):
        # Ensure the docs folder exists
        if not os.path.exists("docs"):
            os.makedirs("docs")

        st.set_page_config(page_title="AI Document Workspace", page_icon="🤖", layout="wide")

        # Multi-session state initialization
        if "chats" not in st.session_state:
            st.session_state.chats = {}  # {session_id: {"name":..., "messages":[], "vector_db":None}}
        if "active_chat_id" not in st.session_state:
            st.session_state.active_chat_id = None
        
        # Recover sessions from docs and FAISS indexes on startup
        self.recover_sessions()

    def recover_sessions(self):
        """
        Recover sessions from previously uploaded files and their FAISS indexes
        """
        if not os.path.exists("docs"):
            return
        
        docs_files = os.listdir("docs")
        existing_file_sessions = {chat_data.get("file"): chat_id 
                                  for chat_id, chat_data in st.session_state.chats.items()}
        
        # For each file in docs that doesn't have a session, create one
        for file in docs_files:
            if file not in existing_file_sessions:
                session_id = str(uuid.uuid4())
                vectordb = None
                messages = []
                
                # Try to load FAISS index from metadata-matched file
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
                                            except Exception as e:
                                                pass
                                        msg_path = os.path.join("faiss_indexes", f"{old_session_id}_messages.json")
                                        if os.path.exists(msg_path):
                                            try:
                                                with open(msg_path, "r") as mf:
                                                    messages = json.load(mf)
                                            except Exception:
                                                messages = []
                                        break
                            except Exception:
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
        except Exception:
            pass

    def run(self):
        # --- 1. SIDEBAR: Multi-Session Chat Management ---
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=48)
            st.title("Workspace")
            if st.button("➕ New Chat Session", use_container_width=True, type="primary"):
                new_id = str(uuid.uuid4())
                st.session_state.chats[new_id] = {"name": "New Chat", "messages": [], "vector_db": None, "file": None}
                st.session_state.active_chat_id = new_id
                st.rerun()
            st.write("---")
            st.caption("Recent Conversations")
            to_delete = None
            confirm_delete = None
            for chat_id, chat_data in st.session_state.chats.items():
                cols = st.columns([0.8, 0.2])
                label = chat_data["name"]
                if cols[0].button(label, key=chat_id, use_container_width=True):
                    st.session_state.active_chat_id = chat_id
                    st.rerun()
                # Delete button (trash icon)
                if cols[1].button("🗑️", key=f"delete_{chat_id}", use_container_width=True):
                    confirm_delete = chat_id
            # Confirmation popup for delete
            if confirm_delete:
                st.session_state["show_confirm_delete"] = confirm_delete
            if "show_confirm_delete" in st.session_state:
                chat_id = st.session_state["show_confirm_delete"]
                st.warning(f"Are you sure you want to delete session '{st.session_state.chats[chat_id]['name']}'?", icon="⚠️")
                c1, c2 = st.columns(2)
                if c1.button("Yes, delete", key="yes_delete"):
                    # Clean up file from docs folder if it exists
                    file_to_remove = st.session_state.chats[chat_id].get("file")
                    if file_to_remove:
                        file_path = os.path.join("docs", file_to_remove)
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                            except Exception as e:
                                st.warning(f"Could not delete file: {file_to_remove} ({e})")
                    # Clean up FAISS index file if exists
                    faiss_path = os.path.join("faiss_indexes", f"{chat_id}.faiss")
                    if os.path.exists(faiss_path):
                        try:
                            os.remove(faiss_path)
                        except Exception as e:
                            st.warning(f"Could not delete FAISS index: {faiss_path} ({e})")
                    # Clean up metadata file
                    metadata_path = os.path.join("faiss_indexes", f"{chat_id}_metadata.json")
                    if os.path.exists(metadata_path):
                        try:
                            os.remove(metadata_path)
                        except Exception as e:
                            st.warning(f"Could not delete metadata: {metadata_path} ({e})")
                    msg_path = os.path.join("faiss_indexes", f"{chat_id}_messages.json")
                    if os.path.exists(msg_path):
                        try:
                            os.remove(msg_path)
                        except Exception as e:
                            st.warning(f"Could not delete messages: {msg_path} ({e})")
                    del st.session_state.chats[chat_id]
                    if st.session_state.active_chat_id == chat_id:
                        st.session_state.active_chat_id = None
                    del st.session_state["show_confirm_delete"]
                    st.rerun()
                if c2.button("Cancel", key="cancel_delete"):
                    del st.session_state["show_confirm_delete"]
                    st.rerun()

        st.title("🤖 Enterprise RAG Assistant")
        st.subheader("Upload any document and extract insights instantly.")

        # --- 2. Main Chat Area ---
        if st.session_state.active_chat_id:
            current_chat = st.session_state.chats[st.session_state.active_chat_id]

            # File uploader for this session only
            with st.expander(f"📂 Manage Document for {current_chat['name']}", expanded=True):
                # Use a unique key for file_uploader per session and rerun
                if "file_uploader_counter" not in current_chat:
                    current_chat["file_uploader_counter"] = 0
                unique_key = f"{st.session_state.active_chat_id}_{current_chat['file_uploader_counter']}"
                uploaded_file = st.file_uploader(
                    f"Upload file for {current_chat['name']}",
                    type=["pdf", "docx", "txt", "csv", "png", "jpg", "jpeg"],
                    key=unique_key
                )
                if uploaded_file and current_chat["vector_db"] is None:
                    file_path = os.path.join("docs", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Successfully indexed: {uploaded_file.name} ✅")
                    # Create a unique vector DB for this session (tag with session_id)
                    vectordb = get_vectorstore([uploaded_file.name], from_session_state=False, session_id=st.session_state.active_chat_id)
                    current_chat["vector_db"] = vectordb
                    current_chat["name"] = uploaded_file.name
                    current_chat["file"] = uploaded_file.name
                    # Save FAISS index to disk for persistence
                    if not os.path.exists("faiss_indexes"):
                        os.makedirs("faiss_indexes")
                    faiss_path = os.path.join("faiss_indexes", f"{st.session_state.active_chat_id}.faiss")
                    vectordb.save_local(faiss_path)
                    # Save metadata for recovery
                    metadata = {"file": uploaded_file.name, "session_id": st.session_state.active_chat_id}
                    with open(os.path.join("faiss_indexes", f"{st.session_state.active_chat_id}_metadata.json"), "w") as f:
                        json.dump(metadata, f)
                    self.save_messages(st.session_state.active_chat_id)
                    current_chat["file_uploader_counter"] += 1
                    st.rerun()

            st.write("---")

            # Display chat history for this session
            for msg in current_chat["messages"]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            # Load FAISS index from disk if not in memory
            if current_chat["vector_db"] is None and current_chat.get("file"):
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
            # Chat input for this session
            if current_chat["vector_db"]:
                if user_query := st.chat_input(f"Ask a question about {current_chat['name']}..."):
                    with st.chat_message("user"):
                        st.write(user_query)
                    current_chat["messages"].append({"role": "user", "content": user_query})
                    # Call backend RAG code for this session only
                    from utils.chatbot import get_response
                    response, _ = get_response(user_query, [], current_chat["vector_db"])
                    with st.chat_message("assistant"):
                        st.write(response if response else "This is where your actual RAG backend model output will stream or display.")
                    current_chat["messages"].append({"role": "assistant", "content": response})
                    self.save_messages(st.session_state.active_chat_id)
            else:
                st.info("Upload a document to enable chat for this session.")
        else:
            st.info("Click '+ New Chat Session' in the sidebar to upload a file and start a conversation.")

if __name__ == "__main__":
    app = ChatApp()
    app.run()
