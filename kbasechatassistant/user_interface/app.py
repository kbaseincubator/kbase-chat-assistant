import streamlit as st
from kbasechatassistant.assistant.mrkl_bot import MRKL_bot
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
import os
from kbasechatassistant.util.stream_handler import StreamHandler
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig

def main():
    
    image = 'Kbase_Logo.png'
    # Display the image
    st.image(image, width=400)
    # App title
    st.title('KBase Research Assistant')
    
    with st.sidebar:
        # Get the OPENAI key from the user
        OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
        # Check if environment variables are set
        if not OPENAI_API_KEY:
            st.error("Please set the necessary environment variables.")
            return
    if 'gpt_agent' not in st.session_state:
        llm = ChatOpenAI(temperature=0, model="gpt-4", streaming=True)
        st.session_state["gpt_agent"] = MRKL_bot(llm=llm)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]
    
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)
   
    if prompt := st.chat_input(): 
        # Add user message to chat history
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant") and st.spinner("Thinking .."):
            st_callback = StreamlitCallbackHandler(st.container())
            cfg = RunnableConfig()
            cfg["callbacks"] = [st_callback]
            response = st.session_state["gpt_agent"].agent_executor.invoke({"input": prompt},cfg)
            st.write(response['output'])
        st.session_state.messages.append(ChatMessage(role="assistant",content=response['output']))
        
if __name__ == "__main__":
    main()
