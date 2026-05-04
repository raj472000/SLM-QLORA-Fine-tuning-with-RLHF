from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.dataset_loader import load_sft_dataset
from src.inference.model_loader import load_base_model, load_tokenizer

logger = get_logger(__name__)


def build_lora_config(config: dict) -> LoraConfig:
    lora_cfg = config["lora"]

    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )


def train_sft(config_path: str = "configs/train_config.yaml"):
    config = load_config(config_path)

    dataset = load_sft_dataset(config)

    model_cfg = config["model"]
    train_cfg = config["training"]

    tokenizer = load_tokenizer(
        model_name=model_cfg["base_model_name"],
        trust_remote_code=model_cfg["trust_remote_code"]
    )

    model = load_base_model(
        model_name=model_cfg["base_model_name"],
        use_4bit=model_cfg["use_4bit"],
        trust_remote_code=model_cfg["trust_remote_code"]
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = build_lora_config(config)

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to="none",
        remove_unused_columns=False
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=config["tokenizer"]["max_seq_length"],
        peft_config=lora_config,
        args=training_args
    )

    logger.info("Starting QLoRA SFT training")

    trainer.train(
        resume_from_checkpoint=train_cfg.get("resume_from_checkpoint")
    )

    logger.info("Saving SFT adapter")

    trainer.model.save_pretrained(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])


if __name__ == "__main__":
    train_sft()