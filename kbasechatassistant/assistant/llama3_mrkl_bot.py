import sys
import os
from kbasechatassistant.assistant.chatbot import KBaseChatBot
from langchain_core.language_models.llms import LLM
from langchain_openai import OpenAIEmbeddings
from kbasechatassistant.tools.ragchain import create_llama_ret_chain
from kbasechatassistant.embeddings.embeddings import NOMIC_CATALOG_DB_DIR, NOMIC_DOCS_DB_DIR, NOMIC_TUTORIALS_DB_DIR
from langchain.agents import initialize_agent, AgentExecutor, create_react_agent
from langchain.agents import AgentType
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from kbasechatassistant.assistant.prompts import DEFAULT_SYSTEM_PROMPT_TEMPLATE, create_mrkl_prompt
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain import hub
from kbasechatassistant.tools.kgtool_cosine_sim import InformationTool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import tool
from pathlib import Path

class Llama3_MRKL_bot(KBaseChatBot):
    _docs_db_dir: Path
    _tutorial_db_dir:Path
    def __init__(self:"Llama3_MRKL_bot", llm: LLM, system_prompt_template: str = None, cborg_api_key: str = None, docs_db_dir: Path | str = None, catalog_db_dir:Path | str = None, tutorial_db_dir: Path | str = None) -> None:
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
        if catalog_db_dir is not None:
            self._catalog_db_dir = Path(catalog_db_dir)
        else:
            self._catalog_db_dir = NOMIC_CATALOG_DB_DIR

        for db_path in [self._docs_db_dir,self._catalog_db_dir,self._tutorial_db_dir]:
            self.__check_db_directories(db_path)
        self.__init_mrkl()
    def __setup_cborg_api_key(self, cborg_api_key: str) -> None:
        if cborg_api_key is not None:
            self._cborg_key = cborg_api_key
        elif os.environ.get("CBORG_API_KEY"):
            self._cborg_key = os.environ["CBORG_API_KEY"]
        else:
            raise KeyError("Missing environment variable CBORG_API_KEY") 
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
    def __init_mrkl(self: "Llama3_MRKL_bot") -> None:
          #Create tools here
          #Get the prompts
        embeddings = OpenAIEmbeddings(openai_api_key=self._cborg_key, 
                                      openai_api_base="https://api.cborg.lbl.gov/v1", model="lbl/nomic-embed-text") 
        doc_chain = create_llama_ret_chain(llm = self._llm,embeddings_func=embeddings, persist_directory = NOMIC_DOCS_DB_DIR)
        tutorial_chain = create_llama_ret_chain(llm = self._llm,embeddings_func=embeddings, persist_directory = NOMIC_TUTORIALS_DB_DIR)
        catalog_chain = create_llama_ret_chain(llm = self._llm,embeddings_func=embeddings, persist_directory = NOMIC_CATALOG_DB_DIR)
        @tool("KG retrieval tool")   
        def KGretrieval_tool(input: str):
           """This tool has the KBase app Knowledge Graph. Useful for when you need to confirm the existance of KBase applications and their appid, tooltip, version, category and data objects.
           This tool can also be used for finding total number of apps or which data objects are shared between apps.
           The input should always be a KBase app name or data object name and should not include any special characters or version number.
           Do not use this tool if you do not have an app or data object name to search with use the KBase Documentation or Tutorial tools instead
           """
           
           response = self._create_KG_agent().invoke({"input": input})
           #Ensure that the response is properly formatted for the agent to use
           if 'output' in response:
                return response['output']
           return "No response from the tool"
        #prompt = hub.pull("hwchase17/react")
        prompt = create_mrkl_prompt(system_prompt_template=self._system_prompt_template)
        tools = [
        Tool.from_function 
        (
            name="KBase Documentation",
            func=doc_chain.invoke,
            description="""This tool should be used for making app recommendations, designing a recommendation plan or analysis workflow. 
            It searches the Kbase documentation. It is useful for answering questions about how to use KBase applications. 
            It does not contain a list of KBase apps.
            Do not use it to search for KBase app presence.
            Use this for making app recommendations. 
            Useful for when you need to find KBase applications to use for user tasks and how to use them. 
            Input should be a fully formed question."""),
        Tool.from_function 
        (
            name="KBase Catalog",
            func=catalog_chain.invoke,
            description="This has the KBase app catalog. This tool is not for useful for making recommendations. Use it for confirmation after finding an app from the documentaiton or tutorial or for when you need to answer questions about what KBase apps are available and what their app id, data objects, full name and preperties are. Input should be a fully formed question."
        ),
         Tool.from_function 
        (
            name="KBase Tutorials",
            func=tutorial_chain.invoke,
            description="This has the tutorial narratives. Useful for when you need to answer questions about using the KBase platform, apps, and features for establishing a workflow to acheive a scientific goal. Input should be a fully formed question."
        ),
        ]

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent = create_react_agent(
        tools = tools,
        llm = self._llm,
        prompt = prompt,
        #stop_sequence = ["Final Answer"],
        )
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    def _create_KG_agent(self):
        
        tools = [InformationTool(uri=os.environ.get('NEO4J_URI'), user=os.environ.get('NEO4J_USERNAME'), password=os.environ.get('NEO4J_PASSWORD'))]

        llm_with_tools = self._llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful tool that finds information about KBase applications in the Knowledge Graph "
                    "Use the tools provided to you to find KBase apps and related properties."
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
            