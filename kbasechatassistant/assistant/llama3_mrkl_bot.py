import sys
import os
from kbasechatassistant.assistant.chatbot import KBaseChatBot
from langchain_core.language_models.llms import LLM
from kbasechatassistant.tools.ragchain import create_llama_ret_chain
from kbasechatassistant.embeddings.embeddings import HF_CATALOG_DB_DIR, HF_DOCS_DB_DIR
from langchain.agents import initialize_agent, AgentExecutor, create_react_agent
from langchain.agents import AgentType
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain import hub

class Llama3_MRKL_bot(KBaseChatBot):

    def __init__(self:"Llama3_MRKL_bot", llm: LLM) -> None:
        super().__init__(llm)
        self.__init_mrkl()
        
    def __init_mrkl(self: "Llama3_MRKL_bot") -> None:
          #Create tools here
          #Get the prompts 
        doc_chain = create_llama_ret_chain(llm = self._llm, persist_directory = HF_DOCS_DB_DIR )
        prompt = hub.pull("hwchase17/react")
        tools = [
        Tool.from_function 
        (
            name="KBase Documentation",
            func=doc_chain.invoke,
            description="This tool has the KBase documentation. Useful for when you need to answer questions about how to use Kbase applications. Input should be a fully formed question."
        ),]

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent = create_react_agent(
        tools = tools,
        llm = self._llm,
        prompt = prompt,
        #stop_sequence = ["Final Answer"],
        )
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        