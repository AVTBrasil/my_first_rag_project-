"""
Assistente RAG de Políticas Internas do RH
- Streamlit (interface)
- LangChain com LCEL (para orquestração)
- FAISS (banco vetorial)
- HuggingFace (embeddings) + Groq (LLM)
"""

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv(override=True)  # lê GROQ_API_KEY do .env


# ===============================================================
# PIPELINE RAG
# ===============================================================

def carregar_pdf():  # lê o pdf e extrai o texto dele
    pdf_loader = PyPDFLoader("politica_rh.pdf")
    texto = pdf_loader.load()
    return texto


def separar_blocos(texto_pdf):
    separador = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    lista_blocos = separador.split_documents(texto_pdf)
    return lista_blocos


def criar_banco_vetores(lista_blocos):
    ferramenta_embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    banco_vetores = FAISS.from_documents(lista_blocos, ferramenta_embedding)
    return banco_vetores


def criar_chain_agente(banco_vetores):
    template = """Você é um assistente de RH que responde perguntas sobre políticas internas da empresa.
    Use APENAS as informações do contexto abaixo para responder.
    Se não encontrar a resposta, diga que não sabe responder.
    Responda em português do Brasil, de forma clara e objetiva.

    Contexto: {context}

    Pergunta: {question}

    Resposta:"""

    prompt_template = ChatPromptTemplate.from_template(template)

    def criar_contexto(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    buscador_banco = banco_vetores.as_retriever()
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    
    chain = (
        {"context": buscador_banco | criar_contexto, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain


@st.cache_resource
def carregar_chain():
    # roda só uma vez, não a cada mensagem
    texto_pdf = carregar_pdf()
    lista_blocos = separar_blocos(texto_pdf)
    banco_vetores = criar_banco_vetores(lista_blocos)
    return criar_chain_agente(banco_vetores)


# ===============================================================
# INTERFACE
# ===============================================================

st.title("Assistente RH - Políticas Internas")

chain = carregar_chain()

# inicia lista de mensagens
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# exibe as mensagens na tela
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

pergunta = st.chat_input("Pergunte sobre as políticas da empresa...")
if pergunta:
    st.session_state["messages"].append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando..."):
            answer = chain.invoke(pergunta)
            st.write(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})