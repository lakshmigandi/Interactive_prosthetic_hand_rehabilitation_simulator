import os
import sqlite3
import gradio as gr

# PDF Loading
from langchain_community.document_loaders import PyPDFLoader

# Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Database & Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Prompting & Chains
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Tools
from langchain_core.tools import tool

# LLM
from langchain_groq import ChatGroq

# LangGraph Agentic Layer
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# Typing
from typing import TypedDict, Annotated

# ==========================================
# 3. PROSTHETIC MANUAL PDF LOADING (Cloud Relative Path)
# ==========================================
exact_path = "3D Robotic Prosthetic Hand Stimulator Final.pdf"
docs = []

print("Processing neuro-prosthetic rehabilitation manuals...")

if os.path.exists(exact_path):
    try:
        loader = PyPDFLoader(exact_path)
        docs.extend(loader.load())
        print(f"🟢 Success! Loaded {len(docs)} pages.")
    except Exception as e:
        print(f"⚠️ Error: Could not load file from the specified path. Details: {e}")
else:
    print(f"⚠️ Warning: {exact_path} not found in root directory.")

# 4. DOCUMENT CHUNKING
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs) if docs else []
print("Chunks Created:", len(chunks)) 

# 5. EMBEDDINGS
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS VECTOR STORE
print("Generating vector storage engine...")
if chunks:
    vector_db = FAISS.from_documents(chunks, embedding_model)
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 20})
    print("Vector storage engine generated successfully.")
else:
    retriever = None
    print("⚠️ Vector storage skipped: No chunks available.")

# ==========================================
# 6. LLM INITIALIZATION
# ==========================================
# Hugging Face Spaces will supply GROQ_API_KEY from Settings Environment Variables
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_retries=2
)
print("LLM initialized successfully") 

# ==========================================
# 7. ADVANCED PROSTHETIC REHABILITATION PROMPT
# ==========================================
template = """
You are an advanced prosthetic hand rehabilitation engineering assistant specialized in:
* prosthetic hand rehabilitation systems
* robotic prosthetic hand simulators
* upper-limb rehabilitation engineering
* hand motor recovery
* finger movement rehabilitation
* grip training rehabilitation
* prosthetic control workflows
* rehabilitation monitoring
* rehabilitation safety procedures
* tele-rehabilitation support
* rehabilitation engineering architectures
* rehabilitation therapy guidance
* rehabilitation progress tracking
* human-machine interaction workflows
* prosthetic training protocols

Use ONLY the provided rehabilitation manual context.

CRITICAL PRIVACY RULES:
* NEVER reveal personal names.
* NEVER reveal inventor names.
* NEVER reveal author names.
* NEVER reveal developer names.
* NEVER reveal ownership information.
* NEVER reveal biographical information.
* NEVER reveal addresses.
* NEVER reveal phone numbers.
* NEVER reveal email addresses.

* If asked who invented, developed, authored, created, owns, or designed the system, respond ONLY with:
"The rehabilitation system is designed as an advanced prosthetic rehabilitation platform intended for rehabilitation monitoring, therapy guidance, motor recovery support, and rehabilitation assistance."

Instructions:
* Answer technically and professionally.
* Focus on rehabilitation workflows.
* Focus on prosthetic engineering.
* Focus on rehabilitation monitoring.
* Focus on motor recovery guidance.
* Explain prosthetic training procedures clearly.
* Explain rehabilitation workflows systematically.

Context:
{context}

Question:
{question}

Technical Answer:
"""

def format_docs(docs_list):
    return "\n\n".join(doc.page_content for doc in docs_list)

# ==========================================
# 8. KAG RETRIEVAL CHAIN
# ==========================================
if retriever:
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | ChatPromptTemplate.from_template(template)
        | llm
        | StrOutputParser()
    )
    print("KAG retrieval chain initialized successfully.")
else:
    rag_chain = None

# ==========================================
# 9. CLINICAL DATABASE INFRASTRUCTURE
# ==========================================
DB_FILE = "prosthetic_hand_suite.db"

def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS resources")
    cur.execute("DROP TABLE IF EXISTS bookings")
    
    cur.execute("""
    CREATE TABLE resources (
        resource_id INTEGER PRIMARY KEY,
        resource_name TEXT,
        status TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE bookings (
        booking_id INTEGER PRIMARY KEY,
        patient_name TEXT,
        resource_id INTEGER,
        FOREIGN KEY(resource_id) REFERENCES resources(resource_id)
    )
    """)
    cur.executemany("""
    INSERT INTO resources (resource_id, resource_name, status)
    VALUES (?, ?, ?)
    """, [
        (1, "Robotic Prosthetic Hand Rehabilitation Simulator", "available"),
        (2, "Finger Motor Recovery Training Unit", "available"),
        (3, "Grip Strength Rehabilitation Station", "available"),
        (4, "Upper Limb Prosthetic Assessment Lab", "available"),
        (5, "Tele-Rehabilitation Monitoring Center", "available"),
        (6, "Prosthetic Control Training Unit", "available")
    ])
    conn.commit()
    conn.close()

init_db()
print("Prosthetic Rehabilitation Database Initialized Successfully") 

# ==========================================
# 10. AGENTIC TOOLS SETUP
# ==========================================
@tool
def check_availability_tool():
    """Check available prosthetic rehabilitation resources."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT resource_id, resource_name FROM resources WHERE status='available'")
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        return "No rehabilitation resources are currently available."
    result = [f"Resource {rid}: {name}" for rid, name in rows]
    return "\n".join(result)

@tool
def book_resource_tool(patient_name: str, resource_id: int):
    """Book a rehabilitation resource for a patient."""
    return "Resource booking functionality is currently under development."

@tool
def facility_information_tool():
    """Provide prosthetic rehabilitation facility information."""
    return """
Prosthetic Rehabilitation Center
A specialized rehabilitation facility designed to support:
• Prosthetic hand rehabilitation
• Finger motor recovery training
• Grip rehabilitation therapy
• Upper-limb rehabilitation workflows
• Prosthetic control training
"""

tools = [check_availability_tool, book_resource_tool, facility_information_tool]
print("Agentic Tools Initialized Successfully") 

# ==========================================
# 11. AGENT LAYER (LANGGRAPH)
# ==========================================
memory = MemorySaver()
agent = create_react_agent(llm, tools, checkpointer=memory)
print("LangGraph Agent Initialized Successfully") 

# ==========================================
# 12. INTELLIGENT ROUTER
# ==========================================
def prosthetic_suite_router(user_query: str) -> str:
    query = user_query.lower()
    if "book" in query:
        return "Resource booking functionality is currently under development."
    elif "available" in query or "availability" in query:
        return check_availability_tool.invoke({})
    elif any(x in query for x in ["facility", "center", "lab", "rehabilitation center", "monitoring center"]):
        return facility_information_tool.invoke({})
    else:
        if rag_chain:
            return rag_chain.invoke(user_query)
        return "Knowledge base context is currently unavailable."

print("Intelligent Router Initialized Successfully") 

# ==========================================
# 13. GRADIO GRAPHICAL UI RUNTIME
# ==========================================
def prosthetic_chat(user_message, history):
    try:
        response = prosthetic_suite_router(user_message)
        return response
    except Exception as e:
        return f"Core Execution Exception: {str(e)}"

with gr.Blocks(theme="soft", title="Interactive Prosthetic Hand Rehabilitation Simulator") as demo:
    gr.Markdown("""
    # 🦾 Interactive Prosthetic Hand Rehabilitation Simulator (KAG + YUOM)
    ### **Built by: Dr. Lakshmi Gandi**
    """)
    
    gr.ChatInterface(
        fn=prosthetic_chat,
        type="messages",
        description="""
**AI-Powered Prosthetic Rehabilitation Engineering Assistant**

**Capabilities:**
• Prosthetic Hand Rehabilitation Knowledge Retrieval
• Robotic Prosthetic Training Guidance
• Finger Movement Rehabilitation Support
• Grip Training Rehabilitation Workflows
• Prosthetic Control System Explanations
• Rehabilitation Monitoring Support
• Tele-Rehabilitation Guidance
• Rehabilitation Resource Availability
        """
    )
    gr.Markdown("--- \n*Clinical Engineering Framework Layer | Secure Multi-Agent Routing Suite*")

if __name__ == "__main__":
    demo.launch()
