from datasets import load_dataset, DatasetDict
from src.data.validator import validate_sft_dataset, validate_dpo_dataset
from src.data.preprocessor import (
    clean_sft_dataset,
    clean_dpo_dataset,
    format_dolly_example,
    format_hh_rlhf_example
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_dataset(dataset, test_size: float, val_size: float, seed: int) -> DatasetDict:
    first_split = dataset.train_test_split(test_size=test_size, seed=seed)
    train_val = first_split["train"].train_test_split(test_size=val_size, seed=seed)

    return DatasetDict({
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": first_split["test"]
    })


def load_sft_dataset(config: dict) -> DatasetDict:
    dataset_cfg = config["dataset"]

    try:
        logger.info(f"Loading SFT dataset: {dataset_cfg['name']}")
        dataset = load_dataset(dataset_cfg["name"], split=dataset_cfg["split"])
    except Exception as exc:
        raise RuntimeError(f"Failed to load SFT dataset: {exc}")

    dataset = validate_sft_dataset(
        dataset=dataset,
        instruction_col=dataset_cfg["instruction_col"],
        response_col=dataset_cfg["response_col"]
    )

    dataset = clean_sft_dataset(dataset, config)

    dataset = dataset.map(
        lambda example: format_dolly_example(example, config),
        remove_columns=dataset.column_names
    )

    return split_dataset(
        dataset=dataset,
        test_size=dataset_cfg["test_size"],
        val_size=dataset_cfg["val_size"],
        seed=dataset_cfg["seed"]
    )


def load_dpo_dataset(config: dict) -> DatasetDict:
    dataset_cfg = config["dataset"]

    try:
        logger.info(f"Loading DPO dataset: {dataset_cfg['name']}")
        dataset = load_dataset(dataset_cfg["name"], split=dataset_cfg["split"])
    except Exception as exc:
        raise RuntimeError(f"Failed to load DPO dataset: {exc}")

    dataset = validate_dpo_dataset(dataset)
    dataset = clean_dpo_dataset(dataset)

    dataset = dataset.map(
        format_hh_rlhf_example,
        remove_columns=dataset.column_names
    )

    return split_dataset(
        dataset=dataset,
        test_size=dataset_cfg["test_size"],
        val_size=dataset_cfg["val_size"],
        seed=dataset_cfg["seed"]
    )