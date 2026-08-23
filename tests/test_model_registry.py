from gemq.utils.model_utils import NAME_TO_MODEL, ModelType, get_model_info


def test_olmoe_0125_instruct_registry_and_metadata():
    model_name = "allenai/OLMoE-1B-7B-0125-Instruct"

    assert NAME_TO_MODEL[model_name] == ModelType.OLMOE

    model_info = get_model_info(model_name)
    assert model_info.num_layers == 16
    assert model_info.first_k_dense_layers == 0
    assert model_info.num_routed_experts_per_layer == 64
    assert model_info.num_shared_experts_per_layer == 0
    assert model_info.num_experts_per_token == 8


def test_qwen3_30b_a3b_instruct_2507_registry_and_metadata():
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    assert NAME_TO_MODEL[model_name] == ModelType.QWEN3MOE

    model_info = get_model_info(model_name)
    assert model_info.num_layers == 48
    assert model_info.first_k_dense_layers == 0
    assert model_info.num_routed_experts_per_layer == 128
    assert model_info.num_shared_experts_per_layer == 0
    assert model_info.num_experts_per_token == 8
