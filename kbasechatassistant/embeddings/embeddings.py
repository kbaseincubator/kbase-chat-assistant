from pathlib import Path

DEFAULT_CATALOG_DB_DIR: Path = Path(__file__).parent / "vector_db_app_catalog"
DEFAULT_DOCS_DB_DIR: Path = Path(__file__).parent / "vector_db_kbase_docs"
DEFAULT_TUTORIAL_DB_DIR: Path = Path(__file__).parent / "vector_db_tutorials"

HF_CATALOG_DB_DIR: Path = Path(__file__).parent / "HFvector_db_app_catalog"
HF_DOCS_DB_DIR: Path = Path(__file__).parent / "HFvector_db_kbase_docs"

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader, DirectoryLoader

def create_embeddings(input_directory, output_directory, embeddings_function = OpenAIEmbeddings):
    # Load data from the specified directory
    json_dir_loader = DirectoryLoader(input_directory, glob="**/[!.]*.json", loader_cls=JSONLoader, loader_kwargs={"jq_schema" : ".[]", "text_content" : False})
    data = json_dir_loader.load()
    
    # Split text into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
    )
    documents = text_splitter.split_documents(data)
    
    # Initialize embeddings model
    embeddings = embeddings_function()
    
    # Create embeddings
    vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=output_directory)
    
    # Persist the database if new embeddings
    vectordb.persist()


