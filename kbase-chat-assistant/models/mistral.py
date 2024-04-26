from KBaseChatAssistant.models.language_model import LanguageModel
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from pathlib import Path


class mistral(LanguageModel):
    def __init__(self, models_folder:Path, name: str = "Mistral-7B-Instruct-v0.2", chat_template: str = None, device:str='auto'):
    
        super().__init__(models_folder=models_folder, name = name, chat_template = chat_template)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path, device_map = self.device)
        self.pipe = pipeline(
            model = self.model, 
            tokenizer = self.tokenizer,
            # return_full_text=True,  # langchain expects the full text
            task='text-generation',
            do_sample = True,
            # we pass model parameters here too
            #stopping_criteria=stopping_criteria,  # without this model rambles during chat
            temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=2000,  # max number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
            )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)