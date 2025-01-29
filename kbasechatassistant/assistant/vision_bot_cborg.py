import sys
import os
from kbasechatassistant.assistant.chatbot import KBaseChatBot
from kbasechatassistant.assistant.prompts import DEFAULT_PROMPT_VISION
from pathlib import Path
import base64
import openai

class Llama_vision_bot_cborg(KBaseChatBot):
    """Vision bot"""
    def __init__(self:"Llama_vision_bot_cborg", model_name: str, cborg_api_key: str = None) -> None:
        self.__setup_cborg_api_key(cborg_api_key)
        self._system_prompt_template = DEFAULT_PROMPT_VISION
        self._model_name = model_name
        self.__init_vision()

    def __setup_cborg_api_key(self, cborg_api_key: str) -> None:
        if cborg_api_key is not None:
            self._cborg_key = cborg_api_key
        elif os.environ.get("CBORG_API_KEY"):
            self._cborg_key = os.environ["CBORG_API_KEY"]
        else:
            raise KeyError("Missing environment variable CBORG_API_KEY")
    
        
    def __init_vision(self: "Llama_vision_bot_cborg") -> None:
        openai.api_key = self._cborg_key
        openai.api_base = "https://api.cborg.lbl.gov"
        pass
          
    @staticmethod
    def encode_image_to_base64(image_path):
        """Convert an image file to a base64 encoded string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def perform_image_analysis(self, image_path):
        base64_image = self.encode_image_to_base64(image_path)
        client = openai.OpenAI(
        api_key=self._cborg_key, 
        base_url="https://api.cborg.lbl.gov" 
        )
        response = client.chat.completions.create(
        model=self._model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._system_prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpg;base64," + base64_image,
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
        )

        return response.choices[-1].message.content



