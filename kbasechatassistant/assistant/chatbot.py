from langchain_core.language_models.llms import LLM

class KBaseChatBot:
    _llm: LLM
    def __init__(self: "KBaseChatBot",  llm: LLM) -> None:
        self._llm = llm
        