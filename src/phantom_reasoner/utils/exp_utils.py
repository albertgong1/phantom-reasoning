import logging
import os
import subprocess
from datetime import datetime

from trl import ModelConfig

from phantom_reasoner.configs import GRPOScriptArguments

logger = logging.getLogger(__name__)


def get_run_name(
    training_algo_name: str,
    script_args: GRPOScriptArguments,
    model_args: ModelConfig,
    run_flags_str: str = "",
) -> str:
    """
    Returns <run_dir>/<dataset_name>/<model_name>/<training_algo_name>/$USER/<MMDD>__<run_flags_str>

    NOTE: `script_args` must have attributes: `dataset_name`, `run_dir`.
    NOTE: `model_args` must have attribute `model_name_or_path`.
    If `run_flags_str` is not provided, the preceding "__" will be omitted.
    """
    assert hasattr(script_args, "run_dir"), "script_args must have attribute run_dir"
    assert hasattr(script_args, "dataset_name"), "script_args must have attribute dataset_name"
    assert hasattr(model_args, "model_name_or_path"), "model_args must have attribute model_name_or_path"

    user = str(os.getenv("USER"))
    today = datetime.now().strftime("%m%d")
    run_name = (
        f"{script_args.run_dir}/{script_args.dataset_name}/{model_args.model_name_or_path}/"
        + f"{training_algo_name}/{user}/{today}"
    )
    if run_flags_str:
        run_name += f"__{run_flags_str}"
    return run_name


def exclude_flash_attn_from_compile() -> None:
    """
    Wrap the registered FlashAttention functions in torch._dynamo.disable, so torch.compile leaves them out.
    FlashAttention is already a single fused kernel, but with the paged KV cache it indexes a Python list
    by `layer_idx`, so compiling it creates one graph per layer and exhausts dynamo's recompile limit.
    Models look the function up in ALL_ATTENTION_FUNCTIONS on every forward, so replacing it there suffices.
    """
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    for impl in ("flash_attention_2", "flash_attention_3", "flash_attention_4"):
        fn = ALL_ATTENTION_FUNCTIONS._global_mapping.get(impl)
        if fn is not None and not getattr(fn, "_torchdynamo_disable", False):
            ALL_ATTENTION_FUNCTIONS._global_mapping[impl] = torch._dynamo.disable(fn)
    logger.info("*** Excluded FlashAttention from torch.compile ***")


def persist_continuous_batching_manager(model) -> None:
    """
    Make `model.generate_batch` keep its continuous batching manager between calls. TRL calls generate_batch
    with the default persistent_manager=False, which reallocates the paged cache, reruns warmup and recaptures
    CUDA graphs on every generation step (about 3 s per step for Qwen3-0.6B). Prefix-cached KV blocks are
    dropped before each call, like TRL's reset_prefix_cache() for vLLM after a weight sync, because they were
    computed with the previous step's weights. Captured CUDA graphs read the weights by address, so this
    relies on the optimizer updating parameters in place (true for DDP and DeepSpeed ZeRO-1/2, not ZeRO-3).
    """
    orig_generate_batch = model.generate_batch

    def generate_batch(inputs, *args, **kwargs):
        manager = getattr(model, "_cached_continuous_batching_manager", None)
        if manager is not None and manager.batch_processor is not None:
            manager.batch_processor.cache.reset()
        kwargs["persistent_manager"] = True
        return orig_generate_batch(inputs, *args, **kwargs)

    model.generate_batch = generate_batch
    logger.info("*** Continuous batching manager will persist across generation steps ***")


def disable_flash_attn_if_unsupported_glibc(model_args: ModelConfig) -> None:
    """
    Get glibc version from ldd --version. If less than 2.32, set attn_implementation to None
    This is because flash-attn==2.8.2 requires GLIBC 2.32, and Anvil has GLIBC 2.82
    """
    try:
        glibc_version = (
            subprocess.check_output(["ldd", "--version"]).decode("utf-8").split("\n")[0].split(" ")[-1]
        )
        logger.info(f"*** Available GLIBC version: {glibc_version}, required: 2.32 ***")
        # glibc_version is like "2.28"
        if float(glibc_version) < 2.32:
            logger.info("*** Setting attn_implementation to None***")
            model_args.attn_implementation = None
    except Exception as e:
        logger.warning(f"*** Error getting glibc version: {e} ***")
