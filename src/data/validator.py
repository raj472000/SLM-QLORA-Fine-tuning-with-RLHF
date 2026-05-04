from datasets import Dataset


def validate_sft_dataset(
    dataset: Dataset,
    instruction_col: str,
    response_col: str
) -> Dataset:
    required_cols = [instruction_col, response_col]

    for col in required_cols:
        if col not in dataset.column_names:
            raise ValueError(f"Missing required column: {col}")

    return dataset


def validate_dpo_dataset(dataset: Dataset) -> Dataset:
    required_cols = ["chosen", "rejected"]

    for col in required_cols:
        if col not in dataset.column_names:
            raise ValueError(f"Missing required column: {col}")

    return dataset