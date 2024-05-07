import sys
import os
from KBaseChatAssistant.assistant.chatbot import KBaseChatBot
from langchain_core.language_models.llms import LLM
from KBaseChatAssistant.tools.ragchain import create_mistral_ret_chain
from KBaseChatAssistant.embeddings.embeddings import HF_CATALOG_DB_DIR, HF_DOCS_DB_DIR
from langchain.agents import initialize_agent, AgentExecutor, create_json_chat_agent
from langchain.agents import AgentType
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

SYSTEM_PROMPT_TEMPLATE = """
You are a halpful KBase chat assistant desgined to help users determine how to use KBase apps for thier data analysis. 
Each task requires multiple steps that are represented by a markdown code snippet of a json blob.
The json structure should contain the following keys:
thought -> your thoughts
action -> name of a tool
action_input -> parameters to send to the tool

These are the tools you can use: {tool_names}.

These are the tools descriptions:

{tools}

If you have enough information to answer the query use the tool "Final Answer". Its parameters is the solution.
If there is not enough information, keep trying.

"""
HUMAN_PROMP_TEMPLATE = """
Add the word "STOP" after each markdown snippet. Example:

```json
{{"thought": "<your thoughts>",
 "action": "<tool name or Final Answer to give a final answer>",
 "action_input": "<tool parameters or the final output"}}
```
STOP

This is my query="{input}". Write only the next step needed to solve it.
Your answer should be based in the previous tools executions, even if you think you know the answer.
Remember to add STOP after each snippet and tool names can only be {tool_names} or Final Answer.

These were the previous steps given to solve this query and the information you already gathered:
"""
class Mistral_MRKL_bot(KBaseChatBot):

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
            func=create_mistral_ret_chain.run,
            description="This tool has the KBase documentation. Useful for when you need to answer questions about how to use Kbase applications. Input should be a fully formed question."
        ),]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", human),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        memory = ConversationBufferMemory(memory_key="chat_history")
        agent = create_json_chat_agent(
        tools = tools,
        llm = mistral_model_custom,
        prompt = prompt,
        stop_sequence = ["STOP"],
        template_tool_response = "{observation}"
        )
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)
        