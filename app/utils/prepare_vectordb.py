from PIL import Image
import pytesseract
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os


# Universal loader for multiple file types
def extract_documents_from_folder(folder_path, file_list=None):
    """
    Extract documents from a folder using appropriate loaders for each file type.
    """
    loaders = {
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.pdf': PyPDFLoader,
    }
    def loader_cls(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in loaders:
            return loaders[ext](path)
        elif ext in ['.png', '.jpg', '.jpeg']:
            # OCR for images
            class OCRLoader:
                def __init__(self, path):
                    self.path = path
                def load(self):
                    try:
                        img = Image.open(self.path)
                        text = pytesseract.image_to_string(img)
                        from langchain_core.documents import Document
                        return [Document(page_content=text, metadata={"source": self.path})]
                    except Exception as e:
                        return []
            return OCRLoader(path)
        else:
            return TextLoader(path)
    # Only load files in file_list if provided
    if file_list is not None:
        docs = []
        for fname in file_list:
            fpath = os.path.join(folder_path, fname)
            loader = loader_cls(fpath)
            docs.extend(loader.load())
        return docs
    else:
        directory_loader = DirectoryLoader(
            folder_path,
            glob="*.*",
            loader_cls=loader_cls
        )
        return directory_loader.load()

def get_text_chunks(docs):
    """
    Split text into chunks

    Parameters:
    - docs (list): List of text documents

    Returns:
    - chunks: List of text chunks
    """
    # Chunk size is configured to be an approximation to the model limit of 2048 tokens
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800, separators=["\n\n", "\n", " ", ""])
    chunks = text_splitter.split_documents(docs)
    return chunks

def get_vectorstore(file_list=None, from_session_state=False, session_id=None):
    """
    Create or retrieve a vectorstore from documents in the docs folder (universal loader).
    Optionally, tag with session_id for session isolation.
    """
    load_dotenv()
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key="API KEY"
    )
    if not from_session_state:
        # Only load and index the file(s) for this session
        docs = extract_documents_from_folder("docs", file_list=file_list)
        # Optionally tag with session_id for future session isolation
        if session_id:
            for doc in docs:
                if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                    doc.metadata = {}
                doc.metadata['session_id'] = session_id
        chunks = get_text_chunks(docs)
        vectordb = FAISS.from_documents(documents=chunks, embedding=embedding)
        return vectordb
    return None
