import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from base64 import b64encode


system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows: 
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

st.set_page_config(page_title = "RAG-powered Medical Document Chatbot")


page_bg_color = """
<style>
  
   [data-testid="stAppViewContainer"] {
        background-color: #3DA121;
     }

 [data-testid="stAppViewContainer"] .stButton > button {
       background-color: white !important;
       color: black !important;
       border-radius: 8px !important;
       padding: 0.6em 1.2em !important;
       display: block;
       margin: 0 auto;
     }

 
   div [data-testid="stAppViewContainer"] textarea {
        color: white;
        border-radius: 8px !important;
        background-color: #1a73e8;
        font-size: 23px;
        border: 2px solid #ffffff !important;
    } 
    
    div [data-testid="stAppViewContainer"] textarea::placeholder {
            color: black;
    }

     div [data-testid="stSidebar"] .stButton > button {
        background-color: white !important;
        color: black !important;
        border-radius: 8px !important;
        padding: 0.6em 1.2em !important;
        display: block;
        font-size: 20px;
    }

    [data-testid="stSidebar"] {
        background-color: #8B0000  !important;
    }

      button:hover {
        background-color: white; !important;
        color: #ff6600 !important;
        border: 1px solid #ff6600 !important;
        box-shadow: 0 0 12px #ff6600 !important;
        transform: scale(1.05);
        transition: 0.3s ease;
    }

    </style>
"""


st.markdown(page_bg_color, unsafe_allow_html=True)

           
def process_document(uploaded_file: UploadedFile) -> list[Document]:
     # Store uploaded file as a temp file
     with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
          temp_file.write(uploaded_file.read())
          temp_path = temp_file.name #Stores the path before closing

     loader = PyMuPDFLoader(temp_path)
     docs = loader.load()

     os.unlink(temp_path) # deletes after loader is done with it

     text_splitter = RecursiveCharacterTextSplitter(
         chunk_size = 400,
         chunk_overlap = 100,
         separators=["\n\n", "\n", ".", "?", "!", " ", ""],
     )

     return text_splitter.split_documents(docs)

def get_vector_collection() -> chromadb.Collection:
   ollama_ef = OllamaEmbeddingFunction(
      url = "http://localhost:11434/api/embeddings",
      model_name = "nomic-embed-text:latest",
   )

   chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
   return chroma_client.get_or_create_collection(
      name = "rag_app",
      embedding_function = ollama_ef,
      metadata={"hnsw:space": "cosine"}, 
   )

def add_to_vector_collection(all_splits: list[Document], file_name: str):

    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
       documents.append(split.page_content)
       metadatas.append(split.metadata)
       ids.append(f"{file_name}_{idx}")
     
    collection.upsert(
        documents = documents,
        metadatas = metadatas,
        ids = ids,
     )
    st.success("Data added to the vector store!")

def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts = [prompt], n_results=n_results)
    return results 

def call_llm(context: str, prompt: str):
    response = ollama.chat(
       model="llama3.2:3b", 
       stream=True,
       messages=[
          {
             "role": "system",
             "content": system_prompt,
          },
          {
             "role": "user",
             "content": f"Context: {context}, Question: {prompt}",
          },
       ]
    )
    for chunk in response:
       if chunk["done"] is False:
          yield chunk["message"]["content"]
       else:
          break
       
def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
   relevant_text = ""
   relevant_text_ids = []

   encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
   ranks = encoder_model.rank(prompt, documents, top_k=3)
   
   for rank in ranks:
       relevant_text += documents[rank["corpus_id"]]
       relevant_text_ids.append(rank["corpus_id"])
  

   return relevant_text, relevant_text_ids


if __name__ == "__main__":
  with st.sidebar:
  
    uploaded_file = st.file_uploader(
         "**📁 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
    )

    process = st.button(
        "⚡Process",
      )

    if uploaded_file and process:
      normalize_uploaded_file_name = uploaded_file.name.translate(
        str.maketrans({"-": "_", ".": "_", " ": "_"}) 
      )
      all_splits = process_document(uploaded_file)
      add_to_vector_collection(all_splits, normalize_uploaded_file_name)

  def get_base64_image(path):
    with open(path, "rb") as img_file:
        return "data:image/jpeg;base64," + b64encode(img_file.read()).decode()
    
  img = get_base64_image("surgical5_logo.jpg")
  st.markdown(
     f"""
         <div style="text-align: center; margin-bottom: 50px;">
             <img src = "{img}" style = "width: 250px; height: 250px; margin-bottom: 30px;  border: 5px solid blue; border-radius: 15px;">
              <h2 style="white-space: nowrap;"> RAG-powered Medical Document Chatbot </h2>
         </div>
      """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style = "font-size: 25px; font-weight: 500; margin-bottom: 15px; text-align: center;">
         Ask a question related to your medical document: 
    </p>
    """,
  unsafe_allow_html=True,
)

prompt = st.text_area(
     label="prompt_box",           
     height=150,                     
     label_visibility="collapsed",
     key="prompt_area"
)



col1, col2, col3 = st.columns([1, 2, 1])

with col2:
     with stylable_container(
    key="ask_button_style",
    css_styles="""
        button {
            font-size: 20px !important;
            padding: 0.6em 1.2em !important;
            border-radius: 8px !important;
            background-color: #ffffff !important;
            color: black !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
    """
):
          ask = st.button(
         "🔥 Ask",
      )

if ask and prompt:
      results = query_collection(prompt)
      context = results.get("documents")[0]
      relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
      response = call_llm(context=relevant_text, prompt=prompt)
      st.write_stream(response)

      with st.expander("See retrieved documents"):
          st.write(results)

      with st.expander("See most relevant document ids"):
          st.write(relevant_text_ids)
          st.write(relevant_text)