import streamlit as st
import os
import dotenv
import uuid
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage
from rag_methods import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

# --- Check if it's Linux ---
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Define available models
if "AZ_OPENAI_API_KEY" not in os.environ:
    MODELS = [
        "openai/gpt-4o-mini",
        "anthropic/claude-1",
        "anthropic/claude-2",
    ]
else:
    MODELS = ["azure-openai/gpt-4o-mini"]

st.set_page_config(
    page_title="RAG LLM App",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.markdown(
    """<h2 style="text-align: center;">📚🔍 <i> LLM CHATBOT USING RAG </i> 🤖💬</h2>""",
    unsafe_allow_html=True
)

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]

# --- Sidebar Configuration ---
with st.sidebar:
    st.subheader("🔑 API Configuration")

    if "AZ_OPENAI_API_KEY" not in os.environ:
        default_openai_api_key = os.getenv("OPENAI_API_KEY", "")
        with st.expander("🔐 OpenAI API Key"):
            openai_api_key = st.text_input(
                "Enter your OpenAI API Key",
                value=default_openai_api_key,
                type="password",
                key="openai_api_key"
            )

        default_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        with st.expander("🔐 Anthropic API Key"):
            anthropic_api_key = st.text_input(
                "Enter your Anthropic API Key",
                value=default_anthropic_api_key,
                type="password",
                key="anthropic_api_key"
            )
    else:
        openai_api_key, anthropic_api_key = None, None
        st.session_state.openai_api_key = None
        az_openai_api_key = os.getenv("AZ_OPENAI_API_KEY")
        st.session_state.az_openai_api_key = az_openai_api_key

# --- API Key Validation ---
missing_openai = not openai_api_key or "sk-" not in openai_api_key
missing_anthropic = not anthropic_api_key

if missing_openai and missing_anthropic and "AZ_OPENAI_API_KEY" not in os.environ:
    st.warning("⬅️ Please enter an API Key to continue...")
else:
    # Sidebar Settings
    with st.sidebar:
        st.divider()
        st.subheader("🤖 Model Selection")
        
        models = [
            model for model in MODELS 
            if ("openai" in model and not missing_openai) or
               ("anthropic" in model and not missing_anthropic) or
               ("azure-openai" in model)
        ]

        st.selectbox("Select a Model", options=models, key="model")

        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle("Use RAG", value=is_vector_db_loaded, key="use_rag", disabled=not is_vector_db_loaded)

        with cols0[1]:
            st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

        st.subheader("📚 RAG Sources")
        uploaded_files = st.file_uploader(
            "📄 Upload a document",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs",
        )

        if uploaded_files:
            load_doc_to_db()

        st.text_input(
            "🌐 Enter a URL",
            placeholder="https://example.com",
            on_change=load_url_to_db,
            key="rag_url",
        )

        with st.expander(f"📚 Documents in DB ({len(st.session_state.rag_sources)})"):
            st.write(st.session_state.rag_sources)

    # --- Chatbot Logic ---
    model_provider = st.session_state.model.split("/")[0]
    
    if model_provider == "openai":
        llm_stream = ChatOpenAI(
            api_key=openai_api_key,
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    elif model_provider == "anthropic":
        llm_stream = ChatAnthropic(
            api_key=anthropic_api_key,
            model=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    elif model_provider == "azure-openai":
        llm_stream = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
            openai_api_version="2024-02-15-preview",
            model_name=st.session_state.model.split("/")[-1],
            openai_api_key=os.getenv("AZ_OPENAI_API_KEY"),
            openai_api_type="azure",
            temperature=0.3,
            streaming=True,
        )

    # Display Previous Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]

            if not st.session_state.use_rag:
                st.write_stream(stream_llm_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_rag_response(llm_stream, messages)) 