from datasets import Dataset


def clean_text(value) -> str:
    if value is None:
        return ""

    return str(value).strip()


def format_dolly_example(example: dict, config: dict) -> dict:
    instruction_col = config["dataset"]["instruction_col"]
    response_col = config["dataset"]["response_col"]
    context_col = config["dataset"]["context_col"]

    instruction = clean_text(example.get(instruction_col))
    context = clean_text(example.get(context_col))
    response = clean_text(example.get(response_col))

    if context:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Context:\n"
            f"{context}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )

    return {
        "text": prompt + response
    }


def clean_sft_dataset(dataset: Dataset, config: dict) -> Dataset:
    instruction_col = config["dataset"]["instruction_col"]
    response_col = config["dataset"]["response_col"]

    def is_valid(example):
        instruction = clean_text(example.get(instruction_col))
        response = clean_text(example.get(response_col))

        return len(instruction) >= 3 and len(response) >= 3

    dataset = dataset.filter(is_valid)

    seen = set()

    def deduplicate(example):
        instruction = clean_text(example.get(instruction_col)).lower()
        response = clean_text(example.get(response_col)).lower()

        key = instruction + "||" + response

        if key in seen:
            return False

        seen.add(key)
        return True

    return dataset.filter(deduplicate)


def clean_dpo_dataset(dataset: Dataset) -> Dataset:
    def is_valid(example):
        chosen = clean_text(example.get("chosen"))
        rejected = clean_text(example.get("rejected"))

        return len(chosen) > 5 and len(rejected) > 5 and chosen != rejected

    return dataset.filter(is_valid)


def format_hh_rlhf_example(example: dict) -> dict:
    chosen = clean_text(example["chosen"])
    rejected = clean_text(example["rejected"])

    prompt = extract_prompt_from_conversation(chosen)

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }


def extract_prompt_from_conversation(text: str) -> str:
    if "\n\nAssistant:" in text:
        return text.split("\n\nAssistant:")[0]

    return text[:500]