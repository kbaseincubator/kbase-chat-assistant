import pytest
import unittest
from unittest.mock import MagicMock, patch
from streamlit.testing.v1 import AppTest
from unittest import TestCase
from langchain.schema import ChatMessage

#from kbasechatassistant.models.CustomMistral import CustomLLMMistral
# Mocking user input
def mock_user_input(input_text):
    def wrapper(input_element):
        input_element.input(input_text).run()
    return wrapper
    
class MockMRKLBot:
    def __init__(self, *args, **kwargs):
        self.agent_executor = MagicMock()
        self.agent_executor.invoke.return_value = {"output": "Mocked agent response"}

class MockMistralMRKLBot(MockMRKLBot):
    pass

@patch("streamlit.image")  # Mocking the st.image function    
def test_select_model():
    at = AppTest.from_file("../../kbasechatassistant/user_interface/app.py").run(timeout=100)
    # # Find the selectbox element by its label
    # No exceptions were rendered in the app output
    assert not at.exception
    
@patch("streamlit.image")  # Mocking the st.image function
def test_sidebar_elements():
    at = AppTest.from_file("../../kbasechatassistant/user_interface/app.py").run(timeout=100)
    at.sidebar.selectbox[0].select("gpt-4").run()
    assert not at.exception
    at.sidebar.selectbox[0].select("Mistral-7B-Instruct-v0.2").run()
    assert not at.exception
    at.sidebar.button[0].click().run(timeout=200)
    assert not at.exception

@patch("kbasechatassistant.assistant.mistral_mrkl_bot.Mistral_MRKL_bot")
@patch("kbasechatassistant.assistant.mrkl_bot.MRKL_bot")
@patch("streamlit.selectbox")
@patch("streamlit.button")
def test_chat_input(mock_button, mock_selectbox, mock_mrkl_bot, mock_mistral_mrkl_bot):
    at = AppTest.from_file("../../kbasechatassistant/user_interface/app.py").run(timeout=100)
    
    # Select a model without actually loading it
    mock_selectbox.select.return_value.run.return_value = None
    
    # Click the button without actually loading the model
    mock_button.click.return_value.run.return_value = None
    
    # Set up the session state with the GPT-4 agent
    #assert isinstance(at.session_state["agent"]._llm, MockMistralMRKLBot)
    at.session_state["agent"] = mock_mrkl_bot
    # Mock user input
    user_input = "What is the weather today?"
    
    # Mock the chat input interaction
    with patch.object(at.session_state["agent"].agent_executor, 'invoke', return_value={"output": "Mock response from Mistral"}):
        #at.chat_input(user_input).run()
        at.chat_input[0].set_value(user_input).run()
        # Ensure user message is in the session state
        TestCase().assertIn(ChatMessage(role="user", content=user_input), at.session_state["messages"])

        # Ensure the agent's response is correct
        response_message = ChatMessage(role="assistant", content="Mock response from Mistral")
        TestCase().assertIn(response_message, at.session_state["messages"])

    # Verify that the user message is added to the chat history
    assert at.session_state["messages"][-1].content == "Mock response from Mistral"

    # Verify that the assistant responds appropriately
    assert at.session_state["messages"][-2].content == user_input

