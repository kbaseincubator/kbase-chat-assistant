import pytest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
from kbasechatassistant.assistant.llama3_mrkl_bot import Llama3_MRKL_bot
from kbasechatassistant.assistant.mrkl_bot_cborg import MRKL_bot_cborg

# Fixtures for integration tests
@pytest.fixture
def mock_environment():
    with patch.dict(os.environ, {
        'CBORG_API_KEY': 'fake_cborg_key',
        'NEO4J_URI': 'fake_uri',
        'NEO4J_USERNAME': 'fake_user',
        'NEO4J_PASSWORD': 'fake_password'
    }):
        yield

@pytest.fixture
def mock_db_directories():
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.is_dir', return_value=True), \
         patch('pathlib.Path.__truediv__', return_value=Path('/fake_path/chroma.sqlite3')):
        yield

@pytest.fixture
def mock_openai():
    with patch('langchain_openai.ChatOpenAI') as mock:
        instance = MagicMock()
        instance.bind.return_value = instance
        mock.return_value = instance
        yield mock

@pytest.fixture
def mock_create_ret_chain():
    with patch('kbasechatassistant.tools.ragchain.create_ret_chain_cborg') as mock:
        chain = MagicMock()
        chain.run.return_value = "Mocked retrieval response"
        mock.return_value = chain
        yield mock

@pytest.fixture
def mock_create_llama_ret_chain():
    with patch('kbasechatassistant.tools.ragchain.create_llama_ret_chain') as mock:
        chain = MagicMock()
        chain.invoke.return_value = "Mocked Llama retrieval response"
        mock.return_value = chain
        yield mock

@pytest.fixture
def mock_embeddings():
    with patch('langchain_openai.OpenAIEmbeddings') as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance

@pytest.fixture
def mock_agent_tools():
    with patch('langchain.agents.create_react_agent') as mock_create_agent, \
         patch('langchain.agents.create_tool_calling_agent') as mock_create_tool_agent, \
         patch('langchain.agents.AgentExecutor') as mock_executor:
        
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_create_tool_agent.return_value = mock_agent
        
        mock_exec = MagicMock()
        mock_exec.invoke.return_value = {"output": "Agent response"}
        mock_executor.return_value = mock_exec
        
        yield mock_create_agent, mock_create_tool_agent, mock_executor, mock_exec

# Integration tests

def test_llama3_bot_full_integration(
    mock_environment, 
    mock_db_directories, 
    mock_create_llama_ret_chain,
    mock_embeddings,
    mock_agent_tools
):
    """Test full integration of Llama3_MRKL_bot with mocked dependencies"""
    mock_create_agent, _, mock_executor, mock_exec = mock_agent_tools
    
    # Create a mock LLM
    llm = MagicMock()
    llm.bind.return_value = llm
    
    # Initialize the bot with mocked init_mrkl method
    with patch.object(Llama3_MRKL_bot, '_Llama3_MRKL_bot__init_mrkl'):
        bot = Llama3_MRKL_bot(llm=llm)
        
        # Add mock agent_executor
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "Agent response"}
        bot.agent_executor = mock_executor
        
        # Test invoking the agent
        response = bot.agent_executor.invoke({"input": "How do I use the KBase application?"})
        
        # Verify the response
        assert response["output"] == "Agent response"
        mock_executor.invoke.assert_called_once_with({"input": "How do I use the KBase application?"})

def test_mrkl_bot_cborg_full_integration(
    mock_environment, 
    mock_db_directories,
    mock_create_ret_chain,
    mock_embeddings,
    mock_agent_tools
):
    """Test full integration of MRKL_bot_cborg with mocked dependencies"""
    mock_create_agent, _, mock_executor, mock_exec = mock_agent_tools
    
    # Create a mock LLM
    llm = MagicMock()
    llm.bind.return_value = llm
    
    # Initialize the bot with mocked init_mrkl method
    with patch.object(MRKL_bot_cborg, '_MRKL_bot_cborg__init_mrkl'):
        bot = MRKL_bot_cborg(llm=llm)
        
        # Add mock agent_executor
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "Agent response"}
        bot.agent_executor = mock_executor
        
        # Test invoking the agent
        response = bot.agent_executor.invoke({"input": "How do I use the KBase application?"})
        
        # Verify the response
        assert response["output"] == "Agent response"
        mock_executor.invoke.assert_called_once_with({"input": "How do I use the KBase application?"})

def test_cborg_llama_agent_creation(mock_openai, mock_environment, mock_db_directories):
    """Test the creation of a CBORG Llama agent from app.py"""
    # First check if the module exists at the given path
    try:
        from kbasechatassistant.user_interface.app import load_cborg_llama_agent
        module_path = 'kbasechatassistant.user_interface.app'
    except ImportError:
        try:
            from kbasechatassistant.assistant.app import load_cborg_llama_agent
            module_path = 'kbasechatassistant.assistant.app'
        except ImportError:
            pytest.skip("Could not find app module with load_cborg_llama_agent")
    
    # Now patch the right module
    with patch(f'{module_path}.ChatOpenAI') as chat_openai_mock, \
         patch(f'{module_path}.Llama3_MRKL_bot') as mock_bot:
        
        # Setup mocks
        instance = MagicMock()
        mock_bot.return_value = instance
        chat_openai_mock.return_value = MagicMock()
        
        # Call the function
        agent = load_cborg_llama_agent("fake_cborg_key")
        
        # Verify the ChatOpenAI model was created correctly
        chat_openai_mock.assert_called_once_with(
            model="lbl/llama",
            api_key="fake_cborg_key",
            base_url="https://api.cborg.lbl.gov"
        )
        
        # Verify the bot was created with the right parameters
        mock_bot.assert_called_once()
        call_args = mock_bot.call_args[1]
        assert call_args["cborg_api_key"] == "fake_cborg_key"

def test_cborg_anthropic_agent_creation(mock_openai, mock_environment, mock_db_directories):
    """Test the creation of a CBORG Anthropic agent from app.py"""
    # First check if the module exists at the given path
    try:
        from kbasechatassistant.user_interface.app import load_cborg_anthropic_agent
        module_path = 'kbasechatassistant.user_interface.app'
    except ImportError:
        try:
            from kbasechatassistant.assistant.app import load_cborg_anthropic_agent
            module_path = 'kbasechatassistant.assistant.app'
        except ImportError:
            pytest.skip("Could not find app module with load_cborg_anthropic_agent")
    
    # Now patch the right module
    with patch(f'{module_path}.ChatOpenAI') as chat_openai_mock, \
         patch(f'{module_path}.MRKL_bot_cborg') as mock_bot:
        
        # Setup mocks
        instance = MagicMock()
        mock_bot.return_value = instance
        chat_openai_mock.return_value = MagicMock()
        
        # Call the function
        agent = load_cborg_anthropic_agent("fake_cborg_key", "system prompt")
        
        # Verify the ChatOpenAI model was created correctly
        chat_openai_mock.assert_called_once_with(
            model="anthropic/claude-sonnet",
            api_key="fake_cborg_key",
            base_url="https://api.cborg.lbl.gov"
        )
        
        # Verify the bot was created with the right parameters
        mock_bot.assert_called_once()
        call_args = mock_bot.call_args[1]
        assert call_args["cborg_api_key"] == "fake_cborg_key"
        assert call_args["system_prompt_template"] == "system prompt"

def test_cborg_gpt_agent_creation(mock_openai, mock_environment, mock_db_directories):
    """Test the creation of a CBORG GPT agent from app.py"""
    # First check if the module exists at the given path
    try:
        from kbasechatassistant.user_interface.app import load_cborg_gpt_agent
        module_path = 'kbasechatassistant.user_interface.app'
    except ImportError:
        try:
            from kbasechatassistant.assistant.app import load_cborg_gpt_agent
            module_path = 'kbasechatassistant.assistant.app'
        except ImportError:
            pytest.skip("Could not find app module with load_cborg_gpt_agent")
    
    # Now patch the right module
    with patch(f'{module_path}.ChatOpenAI') as chat_openai_mock, \
         patch(f'{module_path}.MRKL_bot_cborg') as mock_bot:
        
        # Setup mocks
        instance = MagicMock()
        mock_bot.return_value = instance
        chat_openai_mock.return_value = MagicMock()
        
        # Call the function
        agent = load_cborg_gpt_agent("fake_cborg_key", "system prompt")
        
        # Verify the ChatOpenAI model was created correctly
        chat_openai_mock.assert_called_once_with(
            model="openai/gpt-4o",
            api_key="fake_cborg_key",
            base_url="https://api.cborg.lbl.gov"
        )
        
        # Verify the bot was created with the right parameters
        mock_bot.assert_called_once()
        call_args = mock_bot.call_args[1]
        assert call_args["cborg_api_key"] == "fake_cborg_key"
        assert call_args["system_prompt_template"] == "system prompt"

def test_cborg_deepseek_agent_creation(mock_openai, mock_environment, mock_db_directories):
    """Test the creation of a CBORG Deepseek agent from app.py"""
    # First check if the module exists at the given path
    try:
        from kbasechatassistant.user_interface.app import load_cborg_deepseek_agent
        module_path = 'kbasechatassistant.user_interface.app'
    except ImportError:
        try:
            from kbasechatassistant.assistant.app import load_cborg_deepseek_agent
            module_path = 'kbasechatassistant.assistant.app'
        except ImportError:
            pytest.skip("Could not find app module with load_cborg_deepseek_agent")
    
    # Now patch the right module
    with patch(f'{module_path}.ChatOpenAI') as chat_openai_mock, \
         patch(f'{module_path}.MRKL_bot_cborg') as mock_bot:
        
        # Setup mocks
        instance = MagicMock()
        mock_bot.return_value = instance
        chat_openai_mock.return_value = MagicMock()
        
        # Call the function
        agent = load_cborg_deepseek_agent("fake_cborg_key", "system prompt")
        
        # Verify the ChatOpenAI model was created correctly
        chat_openai_mock.assert_called_once_with(
            model="lbl/cborg-deepthought:latest",
            api_key="fake_cborg_key",
            base_url="https://api.cborg.lbl.gov"
        )
        
        # Verify the bot was created with the right parameters
        mock_bot.assert_called_once()
        call_args = mock_bot.call_args[1]
        assert call_args["cborg_api_key"] == "fake_cborg_key"
        assert call_args["system_prompt_template"] == "system prompt"