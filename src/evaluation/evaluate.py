import math
import torch
import evaluate

from src.utils.config import load_config
from src.data.dataset_loader import load_sft_dataset
from src.inference.model_loader import load_model_with_adapter
from src.inference.generator import generate_text


def compute_perplexity(model, tokenizer, texts: list[str]) -> float:
    losses = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                labels=inputs["input_ids"]
            )

        losses.append(outputs.loss.item())

    avg_loss = sum(losses) / max(len(losses), 1)

    return math.exp(avg_loss)


def evaluate_model(
    config_path: str = "configs/train_config.yaml",
    adapter_path: str = "outputs/dpo_adapter"
):
    config = load_config(config_path)

    dataset = load_sft_dataset(config)

    model, tokenizer = load_model_with_adapter(
        base_model_name=config["model"]["base_model_name"],
        adapter_path=adapter_path,
        use_4bit=config["model"]["use_4bit"],
        trust_remote_code=config["model"]["trust_remote_code"]
    )

    test_texts = dataset["test"]["text"][:20]

    perplexity = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts
    )

    rouge = evaluate.load("rouge")

    predictions = []
    references = []

    for text in test_texts:
        if "### Response:" not in text:
            continue

        prompt = text.split("### Response:")[0] + "### Response:"
        reference = text.split("### Response:")[-1].strip()

        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=128
        )

        predictions.append(result["output"])
        references.append(reference)

    rouge_score = rouge.compute(
        predictions=predictions,
        references=references
    )

    return {
        "perplexity": perplexity,
        "rouge": rouge_score,
        "samples_evaluated": len(predictions)
    }


if __name__ == "__main__":
    result = evaluate_model()
    print(result)