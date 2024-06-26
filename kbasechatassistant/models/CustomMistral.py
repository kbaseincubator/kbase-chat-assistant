from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Mapping, Any
from pathlib import Path
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

class CustomLLMMistral(LLM):
    model: MistralForCausalLM
    tokenizer: LlamaTokenizerFast

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:

        messages = [
         {"role": "user", "content": prompt},
        ]

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.model.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=512, do_sample=True, pad_token_id=self.tokenizer.eos_token_id, top_k=4, temperature=0.7)
        decoded = self.tokenizer.batch_decode(generated_ids)

        output = decoded[0].split("[/INST]")[1].replace("</s>", "").strip()

        if stop is not None:
          for word in stop:
            output = output.split(word)[0].strip()

        while not output.endswith("```"):
          output += "`"

        return output

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

