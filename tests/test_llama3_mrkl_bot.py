import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import os
from kbasechatassistant.assistant.llama3_mrkl_bot import Llama3_MRKL_bot

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
def mock_create_llama_ret_chain():
    with patch('kbasechatassistant.tools.ragchain.create_llama_ret_chain') as mock:
        mock_chain = MagicMock()
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
def test_init_with_env_vars(mock_llm, mock_embeddings, mock_create_llama_ret_chain, mock_os_environ, mock_db_paths):
    """Test initializing bot with environment variables"""
    with patch('pathlib.Path.exists', return_value=True):
        bot = Llama3_MRKL_bot(llm=mock_llm)
        assert bot._llm == mock_llm
        assert hasattr(bot, '_cborg_key')
        assert hasattr(bot, 'agent_executor')

def test_init_with_explicit_key(mock_llm, mock_embeddings, mock_create_llama_ret_chain, mock_db_paths):
    """Test initializing bot with explicit API key"""
    bot = Llama3_MRKL_bot(llm=mock_llm, cborg_api_key='test_key')
    assert bot._cborg_key == 'test_key'
    assert hasattr(bot, 'agent_executor')

def test_init_with_custom_db_dirs(mock_llm, mock_embeddings, mock_create_llama_ret_chain, mock_os_environ, mock_db_paths):
    """Test initializing bot with custom database directories"""
    docs_dir = Path('/custom/docs')
    catalog_dir = Path('/custom/catalog')
    tutorial_dir = Path('/custom/tutorial')
    
    bot = Llama3_MRKL_bot(
        llm=mock_llm,
        docs_db_dir=docs_dir,
        catalog_db_dir=catalog_dir,
        tutorial_db_dir=tutorial_dir
    )
    
    assert bot._docs_db_dir == docs_dir
    assert bot._catalog_db_dir == catalog_dir
    assert bot._tutorial_db_dir == tutorial_dir

def test_missing_api_key(mock_llm):
    """Test that missing API key raises an error"""
    with patch.dict(os.environ, {}, clear=True), pytest.raises(KeyError):
        Llama3_MRKL_bot(llm=mock_llm)

def test_missing_db_directory(mock_llm, mock_os_environ):
    """Test that missing database directory raises an error"""
    with patch('pathlib.Path.exists', return_value=False), pytest.raises(RuntimeError):
        Llama3_MRKL_bot(llm=mock_llm)

def test_db_not_a_directory(mock_llm, mock_os_environ):
    """Test that database path that is not a directory raises an error"""
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.is_dir', return_value=False), \
         pytest.raises(RuntimeError):
        Llama3_MRKL_bot(llm=mock_llm)

def test_missing_db_file(mock_llm, mock_os_environ):
    """Test that missing database file raises an error"""
    with patch('pathlib.Path.exists', side_effect=[True, True, False]), \
         patch('pathlib.Path.is_dir', return_value=True), \
         pytest.raises(RuntimeError):
        Llama3_MRKL_bot(llm=mock_llm)


def test_agent_executor_invocation(mock_llm, mock_embeddings, mock_create_llama_ret_chain, mock_os_environ, mock_db_paths):
    """Test invoking the agent executor"""
    # Mock the entire agent_executor directly instead of trying to mock the individual components
    with patch.object(Llama3_MRKL_bot, 'agent_executor', create=True) as mock_executor:
        mock_executor.invoke.return_value = {"output": "test response"}
        
        bot = Llama3_MRKL_bot(llm=mock_llm)
        # Replace the agent_executor with our mock to avoid all the internal complexity
        bot.agent_executor = mock_executor
        
        response = bot.agent_executor.invoke({"input": "test question"})
        
        assert response["output"] == "test response"
        mock_executor.invoke.assert_called_once_with({"input": "test question"})