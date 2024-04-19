import os
import io
import torch
import onnxruntime
from config import config
from utils.encryption.main import load_encode_file, generate_encryption_key

MODEL_IS_ENCODE = config.ENCODED_MODELS

def torch_load_content(model: str):
    model_path = os.path.abspath(model)
    assert os.path.exists(model_path) is True, f"{model} is not existed"
    print("[torch_load_content]", model)
    if not MODEL_IS_ENCODE:
        return model_path
    encode_binary = load_encode_file(model_path, generate_encryption_key(config.PUBLIC_KEY))
    # print("[torch_load_content]", model)
    return io.BytesIO(encode_binary)


def onnx_model_inference(model: str,device = 'gpu'):
    model_path = os.path.abspath(model)
    assert os.path.exists(model_path) is True, f"{model} is not existed"
    cpu = config.DEVICE == 'cpu'
    cuda = not cpu and torch.cuda.is_available()
    print(cuda)
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # session_options.add_session_config_entry('session.dynamic_block_base', '4')
    if device=='gpu':
        execution_providers = ["ROCMExecutionProvider","CPUExecutionProvider"]
    else:
        execution_providers = ["CPUExecutionProvider"]
    # if cuda:
    #     cuda_provider_options = {
    #         "arena_extend_strategy": "kSameAsRequested",
    #         "cudnn_conv_algo_search": "DEFAULT",
    #     }
    #     execution_providers = [
    #         ("CUDAExecutionProvider", cuda_provider_options),
    #         "CPUExecutionProvider",
    #     ]
    # print("[onnx_model_load] InferenceSession", model, execution_providers)
    model_content = model_path
    # if MODEL_IS_ENCODE:
    #     model_content = load_encode_file(model_path, generate_encryption_key(config.PUBLIC_KEY))
    session = onnxruntime.InferenceSession(
        model_content, session_options, providers=execution_providers
    )
    print(session.get_providers())
    return session
    
    
