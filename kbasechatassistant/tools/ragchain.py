
from langchain_core.language_models.llms import LLM
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain.memory import ReadOnlySharedMemory, ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#HF_EMBEDDING_MODEL_PATH: Path = Path("/Users/prachigupta/LLM/kbase-chat-assistant/embedding_models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/MiniLM-L6-v2")
#HF_EMBEDDING_MODEL_PATH: Path = Path("/app/embedding_models/MiniLM-L6-v2")
#Retrieval chain with openai embeddings, requires openai key
def create_ret_chain(llm: LLM, openai_key: str, persist_directory: str | Path) -> RetrievalQA:
    # Embedding functions to use
    embeddings = OpenAIEmbeddings(openai_api_key = openai_key)
    # Use the persisted database
    vectordb = Chroma(
        persist_directory=str(persist_directory), embedding_function=embeddings
    )
    retriever = vectordb.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    chain_type = "refine"

    # Retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = chain_type,
        retriever = retriever,
        memory = readonlymemory,
    )
    return qa_chain
#For cborg use HF or Nomic embeddings, no openai key required
def create_ret_chain_cborg(llm: LLM,embeddings_func, persist_directory: str | Path) -> RetrievalQA:
    # Embedding functions to use
    #HFembeddings = HuggingFaceEmbeddings(model_name=str(HF_EMBEDDING_MODEL_PATH))
    # Use the persisted database
    vectordb = Chroma(
        persist_directory=str(persist_directory), embedding_function=embeddings_func
    )
    retriever = vectordb.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    chain_type = "refine"

    # Retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = chain_type,
        retriever = retriever,
        memory = readonlymemory,
    )
    return qa_chain

def create_llama2_ret_chain(llm, persist_directory: str | Path) -> RetrievalQA:
    prefix_from_user = "You are an expert in usage of KBase"
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<>\n", "\n<>\n\n"
    system_prompt = B_SYS+ prefix_from_user +E_SYS
    template = """[INST] 
    Please always make sure to include quality control in your guidance. Use the following pieces of retrieved context to answer the question :
    {context}
    {question} [/INST]
    """
    prompt = PromptTemplate(template=system_prompt+template, input_variables=["question","context"])
    # Embedding functions to use
    HFembeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Use the persisted database
    vectordb = Chroma(
        persist_directory=str(persist_directory), embedding_function=HFembeddings
    )
    retriever = vectordb.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    chain_type = "stuff"

    # Retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = chain_type,
        retriever = retriever,
        memory = readonlymemory,
        #chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain
def create_mistral_ret_chain(llm, persist_directory: str | Path) -> RetrievalQA:
    prompt_template = """
    ### [INST] 
    Instruction: Answer the question based on the provided context:
    
    {context}
    
    ### QUESTION:
    {question} 
    
    [/INST]
     """

    # Create prompt from prompt template 
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    
    # Embedding functions to use
    HFembeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Use the persisted database
    vectordb = Chroma(
        persist_directory=str(persist_directory), embedding_function=HFembeddings
    )
    retriever = vectordb.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    chain_type = "stuff"
    
    # Retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = chain_type,
        retriever = retriever,
        memory = readonlymemory,
        #chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def create_llama_ret_chain(llm : LLM,embeddings_func, persist_directory: str | Path) -> RetrievalQA:
    
    
    # Prompt
    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
    )

    # Embedding functions to use
    #HFembeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Use the persisted database
    vectordb = Chroma(
        persist_directory=str(persist_directory), embedding_function=embeddings_func
    )
    retriever = vectordb.as_retriever()
    
    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
