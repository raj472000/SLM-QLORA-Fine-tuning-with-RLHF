from peft import PeftModel
from transformers import TrainingArguments
from trl import DPOTrainer

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.dataset_loader import load_dpo_dataset
from src.inference.model_loader import load_base_model, load_tokenizer

logger = get_logger(__name__)


def train_dpo(config_path: str = "configs/dpo_config.yaml"):
    config = load_config(config_path)

    dataset = load_dpo_dataset(config)

    model_cfg = config["model"]
    dpo_cfg = config["dpo"]

    tokenizer = load_tokenizer(
        model_name=model_cfg["base_model_name"],
        trust_remote_code=model_cfg["trust_remote_code"]
    )

    model = load_base_model(
        model_name=model_cfg["base_model_name"],
        use_4bit=model_cfg["use_4bit"],
        trust_remote_code=model_cfg["trust_remote_code"]
    )

    logger.info("Loading SFT adapter before DPO alignment")

    model = PeftModel.from_pretrained(
        model,
        model_cfg["sft_adapter_path"],
        is_trainable=True
    )

    training_args = TrainingArguments(
        output_dir=dpo_cfg["output_dir"],
        num_train_epochs=dpo_cfg["num_train_epochs"],
        per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=dpo_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        learning_rate=dpo_cfg["learning_rate"],
        logging_steps=dpo_cfg["logging_steps"],
        save_steps=dpo_cfg["save_steps"],
        eval_steps=dpo_cfg["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to="none",
        remove_unused_columns=False
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=dpo_cfg["beta"],
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        max_prompt_length=dpo_cfg["max_prompt_length"],
        max_length=dpo_cfg["max_length"]
    )

    logger.info("Starting DPO training")

    trainer.train()

    logger.info("Saving DPO adapter")

    trainer.model.save_pretrained(dpo_cfg["output_dir"])
    tokenizer.save_pretrained(dpo_cfg["output_dir"])


if __name__ == "__main__":
    train_dpo()