# 🚀 SLM QLoRA + DPO Fine-Tuning Pipeline

## 📌 Overview

This project implements a **production-grade Small Language Model (SLM) fine-tuning pipeline** using:

* **QLoRA (Quantized Low-Rank Adaptation)** for efficient supervised fine-tuning
* **DPO (Direct Preference Optimization)** for post-training alignment
* **FastAPI-based inference service** with safety guardrails and fallback routing
* **Monitoring and evaluation modules** for performance tracking

The system is designed to be **scalable, memory-efficient, and deployment-ready**, following industry-level architectural practices.

---

## 🎯 Objectives

* Fine-tune an open-source SLM using **low GPU memory**
* Align model outputs using **human preference data**
* Provide **safe and controlled inference**
* Enable **reproducible training and evaluation**
* Build a **modular, extensible architecture**

---

## 🧠 Key Features

* ✅ QLoRA-based training (4-bit quantization)
* ✅ LoRA adapter-based fine-tuning (no full model updates)
* ✅ DPO alignment using preference datasets
* ✅ Dataset validation, cleaning, and deduplication
* ✅ Safety layer (prompt injection, toxicity, PII detection)
* ✅ Fallback routing (DPO → SFT → Base model)
* ✅ FastAPI inference service
* ✅ Prometheus-compatible metrics
* ✅ Dockerized deployment
* ✅ Unit testing with pytest

---

## 📂 Project Structure

```
slm-qlora-rl/
│
├── configs/                # YAML configuration files
├── src/
│   ├── data/               # Data loading, validation, preprocessing
│   ├── training/           # QLoRA SFT and DPO training scripts
│   ├── evaluation/         # Model evaluation and comparison
│   ├── inference/          # Model loading, generation, safety, fallback
│   ├── api/                # FastAPI endpoints
│   ├── monitoring/         # Metrics (Prometheus)
│   └── utils/              # Config and logging utilities
│
├── tests/                  # Unit tests
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Multi-service orchestration
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 📊 Datasets Used

### Supervised Fine-Tuning (SFT)

* databricks-dolly-15k

  * Instruction-following dataset
  * Used for base model fine-tuning

### Preference Alignment (DPO)

* Anthropic HH-RLHF

  * Contains chosen vs rejected responses
  * Used for alignment training

---

## ⚙️ Installation

### 1. Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```powershell
.venv\Scripts\Activate.ps1
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training Pipeline

### Step 1: Run QLoRA SFT

```bash
python src/training/train_sft.py
```

✔ Trains LoRA adapters on instruction dataset
✔ Uses 4-bit quantized model for memory efficiency

---

### Step 2: Run DPO Alignment

```bash
python src/training/train_dpo.py
```

✔ Aligns model using preference pairs
✔ Improves response quality and safety

---

## 📈 Evaluation

```bash
python src/evaluation/evaluate.py
```

Metrics include:

* Perplexity
* ROUGE Score
* Sample output comparison
* Safety validation

---

## 🌐 Inference API

### Start Server

```bash
cp .env.example .env
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

### Endpoints

#### Health Check

```bash
GET /health
```

#### Model Info

```bash
GET /model-info
```

#### Generate Response

```bash
POST /generate
```

Example:

```bash
curl -X POST http://localhost:8000/generate \
-H "Content-Type: application/json" \
-d '{"prompt":"Explain QLoRA in simple terms"}'
```

---

## 🛡️ Safety Features

* Prompt injection detection
* Unsafe intent filtering
* PII detection (emails, SSNs, etc.)
* Output moderation
* Automatic fallback routing

---

## 🔁 Fallback Strategy

```text
DPO Model → SFT Model → Base Model
```

Ensures:

* Robustness
* High availability
* Safe responses

---

## 📊 Monitoring

Exposed metrics:

* API request count
* Inference latency
* Fallback rate
* Safety rejection rate

Access:

```bash
GET /metrics
```

---

## 🐳 Docker Deployment

```bash
docker compose up --build
```

---

## 🧪 Testing

```bash
pytest
```

---

## ⚖️ Design Tradeoffs

| Decision              | Advantage        | Tradeoff                 |
| --------------------- | ---------------- | ------------------------ |
| QLoRA                 | Low GPU memory   | Slight precision loss    |
| LoRA adapters         | Fast training    | Limited capacity         |
| DPO                   | Stable alignment | Requires preference data |
| 4-bit quantization    | Efficient        | Minor accuracy impact    |
| Adapter-based serving | Flexible         | Slight latency overhead  |

---

## 🚀 Production Enhancements

* Distributed training (DeepSpeed / Accelerate)
* Model registry and versioning
* MLflow / Weights & Biases integration
* Kubernetes deployment
* GPU autoscaling
* Redis caching layer
* Advanced moderation models
* Human evaluation loop

---

## 📌 Conclusion

This project demonstrates a **complete lifecycle of LLM fine-tuning**, from data processing to deployment, with a strong focus on:

* Efficiency
* Safety
* Scalability
* Real-world applicability

---

## 🤝 Contributions

Contributions are welcome. Please follow standard GitHub workflow:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

This project is intended for **research and educational purposes**. Ensure compliance with dataset and model licenses before production use.

---
