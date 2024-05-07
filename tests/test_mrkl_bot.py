# Tests adapted for MRKL agent from https://github.com/kbaseIncubator/narrative_llm_agent/blob/main/tests/agents/test_analyst_agent.py

import sys, os
import pytest

# Set up fake Neo4j authentication credentials for testing
os.environ["NEO4J_URI"] = "fake_uri"
os.environ["NEO4J_USERNAME"] = "fake_username"
os.environ["NEO4J_PASSWORD"] = "fake_password"

from kbasechatassistant.assistant.mrkl_bot import MRKL_bot
from langchain_core.language_models.llms import LLM
from kbasechatassistant.tools.ragchain import create_ret_chain
from kbasechatassistant.embeddings.embeddings import DEFAULT_CATALOG_DB_DIR, DEFAULT_DOCS_DB_DIR
from langchain.agents import Tool, AgentType
from langchain_community.graphs import Neo4jGraph

token = "not_a_token"
FAKE_OPENAI_KEY = "fake_openai_api_key"
FAKE_OPENAI_KEY_ENVVAR = "not_an_openai_key_environment"
OPENAI_KEY = "OPENAI_API_KEY"


class MockLLM(LLM):
    def _call():
        pass

    def _llm_type():
        pass

class MockNeo4jGraph(Neo4jGraph):
    def __init__(self):
        pass

@pytest.fixture
def mock_llm():
    return MockLLM()
@pytest.fixture
def mock_neo4j_graph():
    return MockNeo4jGraph()

@pytest.fixture(autouse=True)
def automock_api_key(monkeypatch):
    monkeypatch.setenv(OPENAI_KEY, FAKE_OPENAI_KEY_ENVVAR)


def test_mrkl_bot_init_with_api_key(mock_llm):
    # Test MRKL_bot initialization with an API key
    bot = MRKL_bot(llm=mock_llm, openai_api_key=FAKE_OPENAI_KEY)
    assert bot._openai_key == FAKE_OPENAI_KEY
    assert bot._uri == "fake_uri"

def test_mrkl_bot_init_missing_api_key(mock_llm, monkeypatch):
    # Test MRKL_bot initialization with missing API key
    if OPENAI_KEY in os.environ:
        monkeypatch.delenv(OPENAI_KEY)
    with pytest.raises(KeyError):
        MRKL_bot(llm=mock_llm)

def test_mrkl_bot_init_with_env_var(mock_llm):
    bot = MRKL_bot(llm = mock_llm)
    assert bot._openai_key == FAKE_OPENAI_KEY_ENVVAR


def test_mrkl_bot_agent_creation(mock_llm):
    # Test MRKL_bot agent creation
    bot = MRKL_bot(llm=mock_llm)
    assert bot.agent_executor is not None
    assert isinstance(bot.agent_executor.tools[0], Tool)


