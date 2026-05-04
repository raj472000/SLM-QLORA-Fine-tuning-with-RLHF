import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_quantization_config(use_4bit: bool = True):
    if not use_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )


def load_tokenizer(model_name: str, trust_remote_code: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_base_model(
    model_name: str,
    use_4bit: bool = True,
    trust_remote_code: bool = True
):
    quantization_config = build_quantization_config(use_4bit)

    logger.info(f"Loading base model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=trust_remote_code
    )

    model.config.use_cache = False

    return model


def load_model_with_adapter(
    base_model_name: str,
    adapter_path: str | None,
    use_4bit: bool = True,
    trust_remote_code: bool = True
):
    tokenizer = load_tokenizer(base_model_name, trust_remote_code)

    model = load_base_model(
        model_name=base_model_name,
        use_4bit=use_4bit,
        trust_remote_code=trust_remote_code
    )

    if adapter_path:
        try:
            logger.info(f"Loading LoRA adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception as exc:
            logger.error(f"Adapter loading failed. Using base model. Error: {exc}")

    model.eval()

    return model, tokenizer