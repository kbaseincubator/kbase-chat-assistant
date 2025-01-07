import sys
import os
from kbasechatassistant.assistant.chatbot import KBaseChatBot
from kbasechatassistant.assistant.prompts import MRKL_PROMPT
from langchain_core.language_models.llms import LLM
from kbasechatassistant.tools.ragchain import create_ret_chain_cborg
from kbasechatassistant.embeddings.embeddings import HF_CATALOG_DB_DIR, HF_DOCS_DB_DIR
from langchain.agents import initialize_agent, Tool, AgentExecutor, load_tools, AgentType, create_react_agent
from kbasechatassistant.tools.information_tool import InformationTool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from pathlib import Path

class MRKL_bot_cborg(KBaseChatBot):
    _openai_key: str
    _docs_db_dir: Path
    _tutorial_db_dir:Path
    def __init__(self:"MRKL_bot_cborg", llm: LLM, cborg_api_key: str = None, docs_db_dir: Path | str = None, tutorial_db_dir: Path | str = None) -> None:
        super().__init__(llm)
        self.__setup_cborg_api_key(cborg_api_key)

        # if tutorial_db_dir is not None:
        #     self._tutorial_db_dir = Path(tutorial_db_dir)
        # else:
        #     self._tutorial_db_dir = DEFAULT_TUTORIAL_DB_DIR
        if docs_db_dir is not None:
            self._docs_db_dir = Path(docs_db_dir)
        else:
            self._docs_db_dir = HF_DOCS_DB_DIR

        for db_path in [self._docs_db_dir]:#[self._tutorial_db_dir, self._docs_db_dir]:
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
        doc_chain = create_ret_chain_cborg(llm = self._llm, persist_directory = HF_DOCS_DB_DIR)
        #tutorial_chain = create_ret_chain(llm = self._llm, openai_key = self._openai_key, persist_directory = DEFAULT_TUTORIAL_DB_DIR )

        @tool("KG retrieval tool", return_direct=True)   
        def KGretrieval_tool(input: str):
           """This tool has the KBase app Knowledge Graph. Useful for when you need to confirm the existance of KBase applications and their tooltip, version, category and data objects.
           This tool can also be used for finding total number of apps or which data objects are shared between apps.
           The input should always be a KBase app name or data object name and should not include any special characters or version number.
           Do not use this tool if you do not have an app or data object name to search with use the KBase Documentation or Tutorial tools instead
           """
           return self._create_KG_agent().invoke({"input": input})['output']
            
        tools = [
        Tool.from_function 
        (
            name="KBase Documentation",
            func=doc_chain.run,
            description="This tool has the KBase documentation. Useful for when you need to find KBase applications to use for user tasks and how to use them. Input should be a fully formed question."
        ),
        # Tool.from_function 
        # (
        #     name="KBase Tutorials",
        #     func=tutorial_chain.run,
        #     description="This has the tutorial narratives. Useful for when you need to answer questions about using the KBase platform, apps, and features for establishing a workflow to acheive a scientific goal. Input should be a fully formed question."
        # ),
        ]
        #KGretrieval_tool]
        memory = ConversationBufferMemory(memory_key="mrkl_chat_history",return_messages=True)
        agent = create_react_agent(llm = self._llm, tools = tools, prompt = MRKL_PROMPT)
    
        # Create an agent executor by passing in the agent and tools
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory = memory, handle_parsing_errors=True)

    
    def _create_KG_agent(self):
        
        tools = [InformationTool()]

        llm_with_tools = self._llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful tool that finds information about KBase applications in the Knowledge Graph "
                    " and recommends them. Use the tools provided to you to find KBase apps and related properties.  If tools require follow up questions, "
                    "make sure to ask the user for clarification. Make sure to include any "
                    "available options that need to be clarified in the follow up questions "
                    "Do only the things the user specifically requested. ",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: _format_chat_history(x["chat_history"])
                if x.get("chat_history")
                else [],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        return agent_executor
        