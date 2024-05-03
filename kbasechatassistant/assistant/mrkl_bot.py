import sys
import os
from kbasechatassistant.assistant.chatbot import KBaseChatBot
from kbasechatassistant.assistant.prompts import MRKL_PROMPT
from langchain_core.language_models.llms import LLM
from kbasechatassistant.tools.ragchain import create_ret_chain
from kbasechatassistant.embeddings.embeddings import DEFAULT_CATALOG_DB_DIR, DEFAULT_DOCS_DB_DIR
from langchain.agents import initialize_agent, Tool, AgentExecutor, load_tools, AgentType, create_react_agent
from kbasechatassistant.tools.information_tool import InformationTool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import tool
from kbasechatassistant.util.neo4j_config import Neo4jConfig

class MRKL_bot(KBaseChatBot):
    _openai_key: str

    def __init__(self:"MRKL_bot", llm: LLM, openai_api_key: str = None, neo4j_conf: Neo4jConfig = None) -> None:
        super().__init__(llm)
        self.__setup_openai_api_key(openai_api_key)
        self.__setup_neo4j(neo4j_conf)
        self.__init_mrkl()
        

    def __setup_openai_api_key(self, openai_api_key: str) -> None:
        if openai_api_key is not None:
            self._openai_key = openai_api_key
        elif os.environ.get("OPENAI_API_KEY"):
            self._openai_key = os.environ["OPENAI_API_KEY"]
        else:
            raise KeyError("Missing environment variable OPENAI_API_KEY")
            
    def __setup_neo4j(self, neo4j_conf: Neo4jConfig) -> None:
        if neo4j_conf is not None:
            self._uri = neo4j_conf.uri
            self._username = neo4j_conf.username
            self._password  = neo4j_conf.password
        elif os.environ.get("NEO4J_URI") and os.environ.get("NEO4J_USERNAME") and os.environ.get("NEO4J_PASSWORD"):
            self._uri = os.environ["NEO4J_URI"]
            self._username = os.environ["NEO4J_USERNAME"]
            self._password  = os.environ["NEO4J_PASSWORD"]
        else:
            raise KeyError("Missing environment variable for Neo4j")
    
        
    def __init_mrkl(self: "MRKL_bot") -> None:
          #Create tools here
          #Get the prompts 
        doc_chain = create_ret_chain(llm = self._llm, openai_key = self._openai_key, persist_directory = DEFAULT_DOCS_DB_DIR )
        @tool("KG retrieval tool", return_direct=True)   
        def KGretrieval_tool(input: str):
           """This tool has the KBase app Knowledge Graph. Useful for when you need to confirm the existance of KBase applications and their tooltip, version, category and data objects.
           This tool can also be used for finding total number of apps or which data objects are shared between apps.
           The input should always be a KBase app name or data object name and should not include any special characters or version number."""
           return self._create_KG_agent().invoke({"input": input})['output']
            
        tools = [
        Tool.from_function 
        (
            name="KBase Documentation",
            func=doc_chain.run,
            description="This tool has the KBase documentation. Useful for when you need to find KBase applications to use for user tasks and how to use them. Input should be a fully formed question."
        ),
        KGretrieval_tool]
        agent = create_react_agent(llm = self._llm, tools = tools, prompt = MRKL_PROMPT)
    
        # Create an agent executor by passing in the agent and tools
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)
        # SUFFIX = """Provide the final answer and terminate the chain of thought once you have an answer. Begin! 

        # Question: {input}
        # Thought:{agent_scratchpad}"""
        # self.agent = initialize_agent(tools, self._llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True, agent_kwargs={'suffix':SUFFIX})
    
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
        