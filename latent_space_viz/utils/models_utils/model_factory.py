from pipeline.model_utils.model_base import ModelBase

def construct_model_base(model_path: str, checkpoint:str) -> ModelBase:

    if 'qwen' in model_path.lower():
        from pipeline.model_utils.qwen_model import QwenModel
        print('qwen')
        print(checkpoint)
        return QwenModel(model_path, checkpoint=checkpoint)
    if 'llama-3' in model_path.lower():
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    elif 'llama' in model_path.lower():
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path)
    elif 'gemma' in model_path.lower():
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path, checkpoint=checkpoint) 
    elif 'apertus' in model_path.lower():
        from pipeline.model_utils.apertus_model import ApertusModel
        return ApertusModel(model_path, checkpoint=checkpoint) 
    elif 'yi' in model_path.lower():
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
