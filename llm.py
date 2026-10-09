from typing import Any

import torch
from langchain_core.language_models.llms import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from settings import LLM_MAX_LENGTH, MODEL_NAME, USE_4BIT

_tokenizer = None
_model = None


def get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if USE_4BIT:
            # 4-bit so the model fits on the free Colab GPU
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, quantization_config=bnb_config, device_map="auto"
            )
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
            )
    return _tokenizer, _model


def generate_text(prompt, max_length=1000, num_return_sequences=1):
    tokenizer, model = get_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs][0]


class CustomHFLLM(LLM):
    def _call(self, prompt: str, stop: Any = None) -> str:
        return generate_text(prompt, max_length=LLM_MAX_LENGTH)

    @property
    def _llm_type(self) -> str:
        return "custom_huggingface"


llm = CustomHFLLM()
