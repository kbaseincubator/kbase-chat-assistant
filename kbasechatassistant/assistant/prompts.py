from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

MRKL_TEMPLATE = '''
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

Check your output and make sure it conforms! Do not output an action and a final answer at the same time. 
Stop after you arrive at the Final Answer. 

Begin!

Question: {input}

Thought:{agent_scratchpad}'''

#MRKL_PROMPT = PromptTemplate.from_template(MRKL_TEMPLATE)

SYSTEM_PROMPT_TEMPLATE = '''
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question


'''
HUMAN_PROMPT_TEMPLATE = '''
Check your output and make sure it conforms! Do not output an action and a final answer at the same time. 
Stop after you arrive at the Final Answer. 

{mrkl_chat_history}
Begin!

Question: {input}

Thought:{agent_scratchpad}

'''

MRKL_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        input_variables=['tools'],
        template=SYSTEM_PROMPT_TEMPLATE
    ),
    MessagesPlaceholder(variable_name='mrkl_chat_history', optional=True),
    HumanMessagePromptTemplate.from_template(
        input_variables=["input", "mrkl_chat_history", "agent_scratchpad"],
        template=HUMAN_PROMPT_TEMPLATE
    )
])
