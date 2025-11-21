import torch
import functools
import os

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

system = "You are Apertus, a helpful assistant created by the SwissAI initiative."
APERTUS_CHAT_TEMPLATE_WITH_SYSTEM = """<s><|system_start|>{system}
Knowledge cutoff: 2024-04
Current date: 2025-10-23<|system_end|><|developer_start|>Deliberation: disabled
Tool Capabilities: disabled<|developer_end|><|user_start|>{instruction}<|user_end|><|assistant_start|>"""

APERTUS_CHAT_TEMPLATE =  """<s><|system_start|>{system}
Knowledge cutoff: 2024-04
Current date: 2025-10-23<|system_end|><|developer_start|>Deliberation: disabled
Tool Capabilities: disabled<|developer_end|><|user_start|>{instruction}<|user_end|><|assistant_start|>"""

def format_instruction_apertus_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        formatted_instruction = APERTUS_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = APERTUS_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_apertus_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_apertus_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_apertus_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def detokenize_instructions_apertus_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_apertus_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_apertus_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    return result

def act_add_apertus_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.transformer.h[layer-1].mlp.c_proj.weight.dtype
    device = model.transformer.h[layer-1].mlp.c_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.transformer.h[layer-1].mlp.c_proj.bias = torch.nn.Parameter(bias)


class ApertusModel():
    def __init__(self, model_name_or_path: str, model_path_lora:str,checkpoint: str):
        self.model_name_or_path = model_name_or_path
        self.model_name_or_path_lora = model_path_lora      
        if self.model_name_or_path_lora !=None: 
            self.model: AutoModelForCausalLM = self._load_model_LoRA(model_name_or_path, model_path_lora, checkpoint)
        else : 
            self.model: AutoModelForCausalLM = self._load_model(model_name_or_path, checkpoint)
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)
        
        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.eoi_toks = self._get_eoi_toks()

        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()

        self.max_batch_size = None

    def del_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model

    def _load_model(self, model_path, checkpoint=False, dtype=torch.bfloat16):
        if checkpoint==False: 
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"),
            ).eval()
        else: 
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"),
                subfolder=f"{checkpoint}",
            ).eval()

        model.requires_grad_(False) 

        return model

    def _load_model_LoRA(self, model_path, model_path_lora, checkpoint=False, dtype=torch.bfloat16):
        # Load base model
        model_base = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,  # Base model
            load_in_4bit=False, 
            dtype=None).eval()
        
        # Load LoRA adapter directly
        model = PeftModel.from_pretrained(
            model_base,
            model_path_lora,
            subfolder=checkpoint 
        ).eval()

        model = model.merge_and_unload()

        model.requires_grad_(False) 

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )
        tokenizer.padding_side = 'left'
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_apertus_chat, tokenizer=self.tokenizer, system=system, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(APERTUS_CHAT_TEMPLATE.split("{instruction}")[-1])
    
    def _get_model_block_modules(self):
        print(self.model)
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_apertus_weights, direction=direction, coeff=coeff, layer=layer)