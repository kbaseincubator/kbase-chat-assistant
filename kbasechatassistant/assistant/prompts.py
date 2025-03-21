from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Default system prompt template (can be overridden)
DEFAULT_SYSTEM_PROMPT_TEMPLATE = """
You are a helpful KBase research assistant. You answer user queries related to the KBase platform and systems biology. 

"""
DEFAULT_PROMPT_VISION = """
You are a vision assistant for the KBase team. Your job is to analayze images. These images are from a KBase app run.  
"""

# Dynamic system prompt support
def create_mrkl_prompt(system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE):
    HUMAN_PROMPT_TEMPLATE = '''
    Answer the following questions accurately and concisely.
    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer

    Thought: you should always think about what to do

    Action: the action to take, should be one of [{tool_names}]

    Action Input: the input to the action

    Observation: the result of the action

    ... (this Thought/Action/Action Input/Observation can repeat N times)

    Thought: I now know the final answer

    Final Answer: the final answer to the original input question.
    Always follow these:
    -Stop after you arrive at the Final Answer. 
     
    - Before proceeding, validate that your output conforms to the required format:
        -Each `Thought:` must be followed by an `Action:` or lead to a `Final Answer:`.
        -Never skip required fields or mix them together.
    -When suggesting apps to user for performing analysis make sure to review the associated meta data and select analysis steps or apps accordingly.
    -When generating detailed multi step analysis plans, be precise suggesting one app per step.
    -Make sure to make recommendations of apps that exist in KBase.
    -Always use the KBase Documentation tool to find relevant KBase apps then check the Knowledge Graph to find the full app name, appid, tooltip, version, category and data objects.
    -Do not use the Knowledge Graph tool if you do not have an app or data object name to search with use the KBase Documentation or Tutorial tools instead.
    Here is the history of the conversation so far:
    {mrkl_chat_history}
    Begin!

    Question: {input}

    Thought:{agent_scratchpad}
    '''

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            template=system_prompt_template
        ),
        MessagesPlaceholder(variable_name='mrkl_chat_history', optional=True),
        HumanMessagePromptTemplate.from_template(
            input_variables=["tools", "input", "mrkl_chat_history", "agent_scratchpad"],
            template=HUMAN_PROMPT_TEMPLATE
        )
    ])

def create_vision_prompt(system_prompt_template: str = DEFAULT_PROMPT_VISION):
    HUMAN_PROMPT_TEMPLATE = """

    The image path for this image is: {image_path}
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

    Begin!
    Here is the conversation so far:
    {chat_history}

    Question: {input}
    Thought:{agent_scratchpad}
    """

    return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                template=system_prompt_template
            ),
            MessagesPlaceholder(variable_name='chat_history', optional=True),
            HumanMessagePromptTemplate.from_template(
                input_variables=["tools", "input", "image_path", "agent_scratchpad"],
                template=HUMAN_PROMPT_TEMPLATE
            )
            ])