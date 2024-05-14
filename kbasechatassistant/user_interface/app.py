import streamlit as st
from kbasechatassistant.assistant.mrkl_bot import MRKL_bot
from kbasechatassistant.assistant.mistral_mrkl_bot import Mistral_MRKL_bot
from kbasechatassistant.models.CustomMistral import CustomLLMMistral
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from pathlib import Path
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
import os
from kbasechatassistant.util.stream_handler import StreamHandler
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig

def load_gpt_agent():
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
    if not OPENAI_API_KEY:
        st.error("Please set the necessary environment variables.")
        return None
    llm = ChatOpenAI(temperature=0, model="gpt-4", streaming=True)
    return MRKL_bot(llm=llm)

def load_mistral_agent():
    name = "Mistral-7B-Instruct-v0.2"
    models_folder = Path("/scratch/ac.pgupta/convLLM/models")
    pretrained_model_name_or_path = str(models_folder / name)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, device_map='cuda')
    mistral_model_custom = CustomLLMMistral(model=model, tokenizer=tokenizer)
    return Mistral_MRKL_bot(mistral_model_custom)
    
def main():
    
    image = 'Kbase_Logo.png'
    # Display the image
    st.image(image, width=400)
    # App title
    st.title('KBase Research Assistant')
    
    with st.sidebar:
        model_choice = st.selectbox("Choose a Model", ["gpt-4", "Mistral-7B-Instruct-v0.2"])
        submit_button = st.button("Submit")
        if submit_button:
            if 'agent' not in st.session_state:
                if model_choice == "gpt-4":
                    st.session_state["agent"] = load_gpt_agent()
                elif model_choice == "Mistral-7B-Instruct-v0.2":
                    st.session_state["agent"] = load_mistral_agent()
            elif model_choice == "gpt-4" and not isinstance(st.session_state["agent"], MRKL_bot):
                st.session_state["agent"] = load_gpt_agent()
            elif model_choice == "Mistral-7B-Instruct-v0.2" and not isinstance(st.session_state["agent"], Mistral_MRKL_bot):
                st.session_state["agent"] = load_mistral_agent()
    
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
            response = st.session_state["agent"].agent_executor.invoke({"input": prompt},cfg)
            st.write(response['output'])
        st.session_state.messages.append(ChatMessage(role="assistant",content=response['output']))
        
if __name__ == "__main__":
    main()
