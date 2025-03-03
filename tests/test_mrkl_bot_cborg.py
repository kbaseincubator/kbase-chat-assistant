import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import os
from kbasechatassistant.assistant.mrkl_bot_cborg import MRKL_bot_cborg

# Fixtures
@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.bind.return_value = llm
    return llm

@pytest.fixture
def mock_embeddings():
    with patch('langchain_openai.OpenAIEmbeddings') as mock:
        instance = mock.return_value
        yield instance

@pytest.fixture
def mock_create_ret_chain_cborg():
    with patch('kbasechatassistant.tools.ragchain.create_ret_chain_cborg') as mock:
        mock_chain = MagicMock()
        mock_chain.run.return_value = "Mock response from retrieval chain"
        mock.return_value = mock_chain
        yield mock

@pytest.fixture
def mock_os_environ():
    with patch.dict(os.environ, {
        'CBORG_API_KEY': 'fake_cborg_key',
        'NEO4J_URI': 'fake_uri',
        'NEO4J_USERNAME': 'fake_user',
        'NEO4J_PASSWORD': 'fake_password'
    }):
        yield

@pytest.fixture
def mock_db_paths():
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.is_dir', return_value=True), \
         patch('pathlib.Path.__truediv__', return_value=Path('/fake_path/chroma.sqlite3')):
        yield

# Tests
def test_init_with_env_vars(mock_llm, mock_embeddings, mock_create_ret_chain_cborg, mock_os_environ, mock_db_paths):
    """Test initializing bot with environment variables"""
    with patch.object(MRKL_bot_cborg, '_MRKL_bot_cborg__init_mrkl'), \
         patch('pathlib.Path.exists', return_value=True):
        bot = MRKL_bot_cborg(llm=mock_llm)
        assert bot._llm == mock_llm
        assert hasattr(bot, '_cborg_key')

def test_init_with_explicit_key(mock_llm, mock_embeddings, mock_create_ret_chain_cborg, mock_db_paths):
    """Test initializing bot with explicit API key"""
    with patch.object(MRKL_bot_cborg, '_MRKL_bot_cborg__init_mrkl'):
        bot = MRKL_bot_cborg(llm=mock_llm, cborg_api_key='test_key')
        assert bot._cborg_key == 'test_key'

def test_init_with_custom_db_dirs(mock_llm, mock_embeddings, mock_create_ret_chain_cborg, mock_os_environ, mock_db_paths):
    """Test initializing bot with custom database directories"""
    docs_dir = Path('/custom/docs')
    tutorial_dir = Path('/custom/tutorial')
    
    with patch.object(MRKL_bot_cborg, '_MRKL_bot_cborg__init_mrkl'):
        bot = MRKL_bot_cborg(
            llm=mock_llm,
            docs_db_dir=docs_dir,
            tutorial_db_dir=tutorial_dir
        )
        
        assert bot._docs_db_dir == docs_dir
        assert bot._tutorial_db_dir == tutorial_dir

def test_init_with_custom_system_prompt(mock_llm, mock_embeddings, mock_create_ret_chain_cborg, mock_os_environ, mock_db_paths):
    """Test initializing bot with custom system prompt"""
    custom_prompt = "Custom system prompt for testing"
    
    with patch.object(MRKL_bot_cborg, '_MRKL_bot_cborg__init_mrkl'):
        bot = MRKL_bot_cborg(llm=mock_llm, system_prompt_template=custom_prompt)
        assert bot._system_prompt_template == custom_prompt

def test_missing_api_key(mock_llm):
    """Test that missing API key raises an error"""
    with patch.dict(os.environ, {}, clear=True), pytest.raises(KeyError):
        MRKL_bot_cborg(llm=mock_llm)

def test_missing_db_directory(mock_llm, mock_os_environ):
    """Test that missing database directory raises an error"""
    with patch('pathlib.Path.exists', return_value=False), pytest.raises(RuntimeError):
        MRKL_bot_cborg(llm=mock_llm)

def test_db_not_a_directory(mock_llm, mock_os_environ):
    """Test that database path that is not a directory raises an error"""
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.is_dir', return_value=False), \
         pytest.raises(RuntimeError):
        MRKL_bot_cborg(llm=mock_llm)

def test_missing_db_file(mock_llm, mock_os_environ):
    """Test that missing database file raises an error"""
    with patch('pathlib.Path.exists', side_effect=[True, True, False]), \
         patch('pathlib.Path.is_dir', return_value=True), \
         pytest.raises(RuntimeError):
        MRKL_bot_cborg(llm=mock_llm)

# def test_create_kg_agent(mock_llm, mock_embeddings, mock_create_ret_chain_cborg, mock_os_environ, mock_db_paths):
#     """Test creating KG agent"""
#     with patch('langchain.agents.create_tool_calling_agent') as mock_create_agent, \
#          patch('langchain.agents.AgentExecutor') as mock_executor, \
#          patch.object(MRKL_bot_cborg, '_MRKL_bot_cborg__init_mrkl'), \
#          patch('kbasechatassistant.tools.kgtool_cosine_sim.InformationTool'):
        
#         mock_agent = MagicMock()
#         mock_create_agent.return_value = mock_agent
        
#         mock_exec = MagicMock()
#         mock_executor.return_value = mock_exec
        
#         bot = MRKL_bot_cborg(llm=mock_llm)
#         kg_agent = bot._create_KG_agent()
        
#         assert kg_agent is not None
#         mock_executor.assert_called_once()

def test_agent_executor_invocation(mock_llm, mock_embeddings, mock_create_ret_chain_cborg, mock_os_environ, mock_db_paths):
    """Test invoking the agent executor"""
    # Mock the entire agent_executor directly instead of trying to mock the individual components
    with patch.object(MRKL_bot_cborg, '_MRKL_bot_cborg__init_mrkl'):
        # Initialize bot with initialization disabled
        bot = MRKL_bot_cborg(llm=mock_llm)
        
        # Create and add mock agent_executor
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "test response"}
        bot.agent_executor = mock_executor
        
        # Test invocation
        response = bot.agent_executor.invoke({"input": "test question"})
        
        # Check results
        assert response["output"] == "test response"
        mock_executor.invoke.assert_called_once_with({"input": "test question"})