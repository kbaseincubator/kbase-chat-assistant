# Tests adapted for MRKL agent from https://github.com/kbaseIncubator/narrative_llm_agent/blob/main/tests/agents/test_analyst_agent.py

import sys, os
import pytest
from kbasechatassistant.assistant.mistral_mrkl_bot import Mistral_MRKL_bot
from langchain_core.language_models.llms import LLM
from langchain.agents import Tool, AgentType
from langchain_community.graphs import Neo4jGraph



class MockLLM(LLM):
    def _call():
        pass

    def _llm_type():
        pass


@pytest.fixture
def mock_llm():
    return MockLLM()


def test_mistral_mrkl_bot_agent_creation(mock_llm):
    # Test MRKL_bot agent creation
    bot = Mistral_MRKL_bot(llm=mock_llm)
    assert bot.agent_executor is not None
    assert isinstance(bot._llm, MockLLM)
    assert isinstance(bot.agent_executor.tools[0], Tool)


