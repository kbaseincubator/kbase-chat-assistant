import sys
import os
from langchain_openai import OpenAIEmbeddings
from kbasechatassistant.assistant.chatbot import KBaseChatBot
from kbasechatassistant.assistant.prompts import DEFAULT_SYSTEM_PROMPT_TEMPLATE, create_mrkl_prompt
from langchain_core.language_models.llms import LLM
from kbasechatassistant.tools.ragchain import create_ret_chain_cborg
#from kbasechatassistant.embeddings.embeddings import HF_CATALOG_DB_DIR, HF_DOCS_DB_DIR, HF_TUTORIALS_DB_DIR
from kbasechatassistant.embeddings.embeddings import NOMIC_CATALOG_DB_DIR, NOMIC_DOCS_DB_DIR, NOMIC_TUTORIALS_DB_DIR
from langchain.agents import Tool, AgentExecutor, create_react_agent, create_tool_calling_agent
from kbasechatassistant.tools.information_tool import InformationTool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from pathlib import Path


class MRKL_bot_cborg(KBaseChatBot):
    _docs_db_dir: Path
    _tutorial_db_dir:Path
    def __init__(self:"MRKL_bot_cborg", llm: LLM, system_prompt_template: str = None, cborg_api_key: str = None, docs_db_dir: Path | str = None, tutorial_db_dir: Path | str = None) -> None:
        super().__init__(llm)
        self.__setup_cborg_api_key(cborg_api_key)
        self._system_prompt_template = system_prompt_template or DEFAULT_SYSTEM_PROMPT_TEMPLATE

        if tutorial_db_dir is not None:
            self._tutorial_db_dir = Path(tutorial_db_dir)
        else:
            self._tutorial_db_dir = NOMIC_TUTORIALS_DB_DIR
        if docs_db_dir is not None:
            self._docs_db_dir = Path(docs_db_dir)
        else:
            self._docs_db_dir = NOMIC_DOCS_DB_DIR

        for db_path in [self._tutorial_db_dir, self._docs_db_dir]:
            self.__check_db_directories(db_path)
        self.__init_mrkl()
        
    def __check_db_directories(self, db_path: Path) -> None:
        """
        Checks for presence of the expected database directory by checking for chroma.sqlite3 file 
        """
        if not db_path.exists():
            raise RuntimeError(
                f"Database directory {db_path} not found, unable to make Agent."
            )
        if not db_path.is_dir():
            raise RuntimeError(
                f"Database directory {db_path} is not a directory, unable to make Agent."
            )
        db_file = db_path / "chroma.sqlite3"
        if not db_file.exists():
            raise RuntimeError(
                f"Database file {db_file} not found, unable to make Agent."
            )   

    def __setup_cborg_api_key(self, cborg_api_key: str) -> None:
        if cborg_api_key is not None:
            self._cborg_key = cborg_api_key
        elif os.environ.get("CBORG_API_KEY"):
            self._cborg_key = os.environ["CBORG_API_KEY"]
        else:
            raise KeyError("Missing environment variable CBORG_API_KEY")
    
        
    def __init_mrkl(self: "MRKL_bot_cborg") -> None:
          #Create tools here
          #Get the prompts 
        embeddings = OpenAIEmbeddings(openai_api_key=self._cborg_key, 
                                      openai_api_base="https://api.cborg.lbl.gov/v1", model="lbl/nomic-embed-text")
        doc_chain = create_ret_chain_cborg(llm = self._llm,embeddings_func=embeddings, persist_directory = NOMIC_DOCS_DB_DIR)
        tutorial_chain = create_ret_chain_cborg(llm = self._llm,embeddings_func=embeddings, persist_directory = NOMIC_TUTORIALS_DB_DIR)

        @tool("KG retrieval tool")   
        def KGretrieval_tool(input: str):
           """This tool has the KBase app Knowledge Graph. Useful for when you need to confirm the existance of KBase applications and their tooltip, version, category and data objects.
           This tool can also be used for finding total number of apps or which data objects are shared between apps.
           The input should always be a KBase app name or data object name and should not include any special characters or version number.
           Do not use this tool if you do not have an app or data object name to search with use the KBase Documentation or Tutorial tools instead
           """
           
           response = self._create_KG_agent().invoke({"input": input})
           #Ensure that the response is properly formatted for the agent to use
           if 'output' in response:
                return response['output']
           return "No response from the tool"
            
        tools = [
        Tool.from_function 
        (
            name="KBase Documentation",
            func=doc_chain.run,
            description="This tool has the KBase documentation. This tool should be the first place to check anything related to KBase and its apps. Use this for making app recommendations. Useful for when you need to find KBase applications to use for user tasks and how to use them. Input should be a fully formed question."
        ),
        Tool.from_function 
        (
            name="KBase Tutorials",
            func=tutorial_chain.run,
            description="This has the tutorial narratives. Useful for when you need to answer questions about using the KBase platform, apps, and features for establishing a workflow to acheive a scientific goal. Input should be a fully formed question."
        ),
        KGretrieval_tool]
        memory = ConversationBufferMemory(memory_key="mrkl_chat_history",return_messages=True)
        prompt = create_mrkl_prompt(system_prompt_template=self._system_prompt_template)

        agent = create_react_agent(llm = self._llm, tools = tools, prompt = prompt)
    
        # Create an agent executor by passing in the agent and tools
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory = memory, handle_parsing_errors=True)

    
    def _create_KG_agent(self):
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful tool that finds information about KBase applications in the Knowledge Graph "
                    "Use the tools provided to you to find KBase apps and related properties."
                    "Do only the things the user specifically requested. ",
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        tools=[InformationTool()]
        agent = create_tool_calling_agent(self._llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        return agent_executor
        