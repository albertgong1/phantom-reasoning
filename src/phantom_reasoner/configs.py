from dataclasses import dataclass, field
from typing import Literal

import trl


@dataclass
class GRPOScriptArguments(trl.ScriptArguments):
    # Train dataset arguments
    training_mode: Literal[
        "pw", "gsminfinite", "rg-family_relationships", "rg-knights_knaves", "hp", "2wiki", "msq"
    ] = "pw"
    dataset_name: str = "kilian-group/phantom-wiki-v1"
    split_list: list[str] = field(
        default_factory=lambda: ["depth_20_size_50_seed_1", "depth_20_size_50_seed_2"]
    )
    from_local: bool = False
    # Eval dataset arguments
    eval_dataset_name: str = "kilian-group/phantom-wiki-v1"
    eval_split_list: str = field(default_factory=lambda: ["depth_20_size_50_seed_3"])
    eval_from_local: bool = False

    # Script arguments
    run_dir: str = "runs"
    reward_func_names: list[str] = field(default_factory=lambda: ["f1"])
    data_curriculum: Literal[
        "random",
        "difficulty_asc",
        "difficulty_desc",
    ] = "random"
    prompt_method: Literal["zeroshot", "cot"] = "cot"
    ignore_think_tags_in_outputs: bool = False
    exclude_aggregation_questions: bool = True
    max_num_train_samples: int = 10_000


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    system_prompt: str | None = field(
        default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    )
    hub_model_revision: str | None = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to overwrite the Hub revision."}
    )
    push_to_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to push to a Hub revision/branch."}
    )
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: str | None = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    compile_exclude_flash_attn: bool = field(
        default=True,
        metadata={
            "help": (
                "Keep FlashAttention out of torch.compile (e.g. transformers continuous batching with "
                "default_compile_level > 0). Compiling it specializes on each layer's layer_idx, which hits "
                "dynamo's recompile limit after a few layers. No effect when nothing is compiled."
            )
        },
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: str | None = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to overwrite the Hub revision."}
    )
    push_to_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to push to a Hub revision/branch."}
    )
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: str | None = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
