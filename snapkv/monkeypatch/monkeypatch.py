from importlib.metadata import version
import warnings
import transformers
from snapkv.monkeypatch.llama_hijack_4_37 import llama_flash_attn2_forward as llama_flash_attn2_forward_4_37, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_4_37
from snapkv.monkeypatch.mistral_hijack_4_37 import mistral_flash_attn2_forward as mistral_flash_attn2_forward_4_37, prepare_inputs_for_generation_mistral as prepare_inputs_for_generation_mistral_4_37
from snapkv.monkeypatch.mixtral_hijack_4_37 import mixtral_flash_attn2_forward as mixtral_flash_attn2_forward_4_37, prepare_inputs_for_generation_mixtral as prepare_inputs_for_generation_mixtral_4_37

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version

def replace_llama():
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_4_37
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_4_37

def replace_mistral():
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_4_37
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_4_37

def replace_mixtral():
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mixtral_4_37
    transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = mixtral_flash_attn2_forward_4_37