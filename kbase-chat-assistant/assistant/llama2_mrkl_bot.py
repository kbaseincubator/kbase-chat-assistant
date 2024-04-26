import sys
import os
from KBaseChatAssistant.assistant.chatbot import KBaseChatBot
from langchain_core.language_models.llms import LLM
from KBaseChatAssistant.tools.ragchain import create_llama2_ret_chain
from KBaseChatAssistant.embeddings.embeddings import HF_CATALOG_DB_DIR, HF_DOCS_DB_DIR
from langchain.agents import initialize_agent, Tool, AgentExecutor, create_react_agent
from langchain.agents import AgentType

class Llama2_MRKL_bot(KBaseChatBot):

    def __init__(self:"Llama2_MRKL_bot", llm: LLM) -> None:
        super().__init__(llm)
        self.__init_mrkl()
        
    def __init_mrkl(self: "Llama2_MRKL_bot") -> None:
          #Create tools here
          #Get the prompts 
        doc_chain = create_llama2_ret_chain(llm = self._llm, persist_directory = HF_DOCS_DB_DIR )
        
        tools = [
        Tool.from_function 
        (
            name="KBase Documentation",
            func=doc_chain.run,
            description="This tool has the KBase documentation. Useful for when you need to answer questions about how to use Kbase applications. Input should be a fully formed question."
        ),]

        SUFFIX = """Provide the final answer and terminate the chain of thought once you have an answer. Begin! 

        Question: {input}
        Thought:{agent_scratchpad}"""
        self.agent = initialize_agent(tools, self._llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True, agent_kwargs={'suffix':SUFFIX})