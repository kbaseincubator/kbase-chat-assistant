from langchain_core.prompts import PromptTemplate

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

MRKL_PROMPT = PromptTemplate.from_template(MRKL_TEMPLATE)