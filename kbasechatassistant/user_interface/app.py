import streamlit as st
from kbasechatassistant.assistant.mrkl_bot import MRKL_bot
from kbasechatassistant.assistant.mrkl_bot_cborg import MRKL_bot_cborg
from kbasechatassistant.assistant.mistral_mrkl_bot import Mistral_MRKL_bot
from kbasechatassistant.assistant.llama3_mrkl_bot import Llama3_MRKL_bot
from kbasechatassistant.models.CustomMistral import CustomLLMMistral
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from pathlib import Path
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
import os
from kbasechatassistant.util.stream_handler import StreamHandler
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig
from kbasechatassistant.assistant.vision_bot_cborg import Llama_vision_bot_cborg

def load_cborg_deepseek_agent(cborg_api_key,system_prompt):
    llm = ChatOpenAI(
    model="lbl/cborg-deepthought:latest",
    api_key=cborg_api_key,
    base_url="https://api.cborg.lbl.gov"  
    )
    return MRKL_bot_cborg(llm=llm,system_prompt_template=system_prompt, cborg_api_key=cborg_api_key)
def load_cborg_vision_agent(cborg_api_key):
    model_name = "lbl/llama-vision"
    return Llama_vision_bot_cborg(model_name=model_name, cborg_api_key=cborg_api_key)
def load_cborg_llama_agent(cborg_api_key):
    llm = ChatOpenAI(
    model="lbl/llama",
    api_key=cborg_api_key,
    base_url="https://api.cborg.lbl.gov"  
    )
    return Llama3_MRKL_bot(llm=llm,cborg_api_key=cborg_api_key)
def load_cborg_gpt_agent(cborg_api_key,system_prompt):
    llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=cborg_api_key,
    base_url="https://api.cborg.lbl.gov"  
    )
    return MRKL_bot_cborg(llm=llm,system_prompt_template=system_prompt, cborg_api_key=cborg_api_key)
def load_cborg_anthropic_agent(cborg_api_key,system_prompt):
    llm = ChatOpenAI(
    model="anthropic/claude-sonnet",
    api_key=cborg_api_key,
    base_url="https://api.cborg.lbl.gov"  
    )
    return MRKL_bot_cborg(llm=llm,system_prompt_template=system_prompt,cborg_api_key=cborg_api_key)

def load_gpt_agent(openai_api_key):
    llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)
    return MRKL_bot(llm=llm, openai_api_key=openai_api_key)

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
    # User inputs for system prompt
    user_system_prompt = st.text_area("System Prompt", value="""You are a helpful assistant designed to assist with KBase tasks. Use the tools and information provided to answer user questions accurately and concisely.
    """, height=200)
    with st.sidebar:
        if "current_model" in st.session_state:
            st.markdown(f"**Active Model:** {st.session_state['current_model']}")
        model_choice = st.selectbox("Choose a Model", ["gpt-4", "Mistral-7B-Instruct-v0.2", "CBORG GPT", "CBORG Llama","CBORG Anthropic","CBORG Deepseek","CBORG Vision" ])
        OPENAI_API_KEY = None
        CBORG_API_KEY = None

        if model_choice == "gpt-4":
            OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
            if not OPENAI_API_KEY:
                st.error("Please enter your OpenAI key")
        if model_choice in ["CBORG GPT", "CBORG Llama","CBORG Anthropic","CBORG Deepseek","CBORG Vision"]:
            CBORG_API_KEY = st.text_input("CBORG API Key", type="password")
            if not CBORG_API_KEY:
                st.error("Please enter your CBORG API key")
                
        submit_button = st.button("Submit")
        
        if submit_button:
             # Check if the model has changed
            if "current_model" not in st.session_state or st.session_state["current_model"] != model_choice:
                st.session_state["current_model"] = model_choice
                st.success(f"Model switched to: {model_choice}")
            if model_choice == "gpt-4" and not OPENAI_API_KEY:
                st.error("Please provide an OpenAI API key and hit the submit button")
                return
            if model_choice in ["CBORG GPT", "CBORG Llama", "CBORG Anthropic","CBORG Deepseek","CBORG Vision"] and not CBORG_API_KEY:
                st.error("Please provide a CBORG API key and hit the submit button")
                return
                
            if 'agent' not in st.session_state or st.session_state["current_model"] != model_choice:
                if model_choice == "gpt-4":
                    st.session_state["agent"] = load_gpt_agent(OPENAI_API_KEY)
                    print(st.session_state["agent"])
                elif model_choice == "Mistral-7B-Instruct-v0.2":
                    st.session_state["agent"] = load_mistral_agent
                elif model_choice == "CBORG GPT":
                    st.session_state["agent"] = load_cborg_gpt_agent(cborg_api_key=CBORG_API_KEY,system_prompt=user_system_prompt)
                    print("cborg gpt agent loaded")
                elif model_choice == "CBORG Llama": 
                    st.session_state["agent"] = load_cborg_llama_agent(CBORG_API_KEY)
                    print("cborg llama agent loaded")
                elif model_choice == "CBORG Anthropic": 
                    st.session_state["agent"] = load_cborg_anthropic_agent(cborg_api_key=CBORG_API_KEY,system_prompt=user_system_prompt)
                    print("cborg Anthropic agent loaded")
                elif model_choice == "CBORG Deepseek": 
                    st.session_state["agent"] = load_cborg_deepseek_agent(cborg_api_key=CBORG_API_KEY,system_prompt=user_system_prompt)
                    print("cborg Anthropic agent loaded")
                elif model_choice == "CBORG Vision":
                    st.session_state["agent"] = load_cborg_vision_agent(CBORG_API_KEY)
                    print("cborg Llama Vision loaded")
            else:
                if model_choice == "gpt-4" and not isinstance(st.session_state["agent"], MRKL_bot):
                    st.session_state["agent"] = load_gpt_agent(OPENAI_API_KEY)
                elif model_choice == "Mistral-7B-Instruct-v0.2" and not isinstance(st.session_state["agent"], Mistral_MRKL_bot):
                    st.session_state["agent"] = load_mistral_agent()
                elif model_choice == "CBORG GPT" and not isinstance(st.session_state["agent"], MRKL_bot_cborg):
                    st.session_state["agent"] = load_cborg_gpt_agent(system_prompt=user_system_prompt, cborg_api_key=CBORG_API_KEY)
                elif model_choice == "CBORG Llama" and not isinstance(st.session_state["agent"], Llama3_MRKL_bot):
                    st.session_state["agent"] = load_cborg_llama_agent(CBORG_API_KEY)
                elif model_choice == "CBORG Anthropic" and not isinstance(st.session_state["agent"], MRKL_bot_cborg):
                    st.session_state["agent"] = load_cborg_anthropic_agent(system_prompt=user_system_prompt, cborg_api_key=CBORG_API_KEY)
                elif model_choice == "CBORG Deepseek" and not isinstance(st.session_state["agent"], MRKL_bot_cborg):
                    st.session_state["agent"] = load_cborg_deepseek_agent(system_prompt=user_system_prompt, cborg_api_key=CBORG_API_KEY)
                elif model_choice == "CBORG Vision" and not isinstance(st.session_state["agent"], Llama_vision_bot_cborg):
                    st.session_state["agent"] = load_cborg_vision_agent(cborg_api_key=CBORG_API_KEY)

    if model_choice == "CBORG Vision":
        uploaded_file = st.file_uploader("Upload an image for analysis (JPEG format)", type=["jpg", "jpeg"])
        if uploaded_file:
            image_path = f"/tmp/{uploaded_file.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Display the uploaded image
            st.image(image_path, caption="Uploaded Image", use_container_width=True)
            with st.spinner("Analyzing image..."):
                response = st.session_state["agent"].perform_image_analysis(image_path)
                
                st.success("Image analyzed successfully!")
                st.markdown(f"**Analysis Output:** {response}")
    else:
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
            
            with st.spinner("Thinking..."):
                assistant_message = st.chat_message("assistant")
                with assistant_message:
                    st_callback = StreamlitCallbackHandler(assistant_message)
                    cfg = RunnableConfig()
                    cfg["callbacks"] = [st_callback]
                    response = st.session_state["agent"].agent_executor.invoke({"input": prompt}, cfg)
                    st.markdown(response['output'])
            st.session_state.messages.append(ChatMessage(role="assistant",content=response['output']))
        
if __name__ == "__main__":
    main()
