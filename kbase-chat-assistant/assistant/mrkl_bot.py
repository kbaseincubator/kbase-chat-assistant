import sys
import os
from KBaseChatAssistant.assistant.chatbot import KBaseChatBot
from langchain_core.language_models.llms import LLM
from KBaseChatAssistant.tools.ragchain import create_ret_chain
from KBaseChatAssistant.embeddings.embeddings import DEFAULT_CATALOG_DB_DIR, DEFAULT_DOCS_DB_DIR
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

class MRKL_bot(KBaseChatBot):
    _openai_key: str

    def __init__(self:"MRKL_bot", llm: LLM, openai_api_key: str = None) -> None:
        super().__init__(llm)
        self.__setup_openai_api_key(openai_api_key)
        self.__init_mrkl()
        

    def __setup_openai_api_key(self, openai_api_key: str) -> None:
        if openai_api_key is not None:
            self._openai_key = openai_api_key
        elif os.environ.get("OPENAI_API_KEY"):
            self._openai_key = os.environ["OPENAI_API_KEY"]
        else:
            raise KeyError("Missing environment variable OPENAI_API_KEY")
        
    def __init_mrkl(self: "MRKL_bot") -> None:
          #Create tools here
          #Get the prompts 
        doc_chain = create_ret_chain(llm = self._llm, openai_key = self._openai_key, persist_directory = DEFAULT_DOCS_DB_DIR )
        
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