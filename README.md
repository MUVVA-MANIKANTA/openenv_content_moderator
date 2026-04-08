---
title: AI Social Guard v2
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags: ["openenv", "reinforcement-learning", "trust-and-safety"]
---

# 🛡️ AI Social Guard v2 (RL + Threat Tracking)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/manikantareddymuvva3/ai-social-guard) ![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)


*   **Multi-Category Moderation**: Automated classification into **HATE, MISINFO, VANDALISM, ADULT, SPAM, and OFFENSIVE** categories.
*   **Serial Offender Tracking**: Proactive risk signaling for repeat violators (e.g., User 999).
*   **Dynamic Reputation Engine**: Context-aware rewards that vary based on the author's history.
*   **AIEvaluation (v2)**: Programmatic graders mapping to OpenEnv Task scoring (0.0 to 1.0).

> **"We are not just moderating content — we are training AI agents using reinforcement learning to improve moderation decisions over time."**
>
> **Note**: This environment exposes specific failure modes of LLM moderation under ambiguity, sarcasm, and context-dependent toxicity, making it a robust benchmark for Trust & Safety agents.

### Real-world Content Moderation Ecosystem for AI Agents

AI Social Guard is a high-fidelity environment built on the **OpenEnv** specification. It simulates the complex task of professional content moderation, where an AI agent must protect online communities by classifying posts into multiple safety categories while considering nuanced factors like **user reputation**.

---

## 🌟 Motivation
In real-world social platforms, moderation isn't just about binary sentiment. It requires distinguishing between sadness and hate speech, identifying subtle misinformation, and vetting suspicious promoters. This environment evaluates an agent's ability to minimize false positives (wrongly flagging trusted users) while maintaining high precision for safety violations.

## 🕹️ Specification

### OpenEnv API
The environment implements the full OpenEnv lifecycle:
- `POST /reset`: Resets the environment to a specific task state.
- `POST /step`: Returns `observation`, `reward`, `done`, and `info`.
- `GET /state`: Returns the full `SocialGuardState` for grading.

### Action Space (`SocialGuardAction`)
- `approve`: Safe for the community.
- `flag_spam`: Promotions, phishing, or bulk advertising.
- `flag_hate`: Slurs, harassment, or dehumanizing speech.
- `flag_adult`: NSFW/Adult content.
- `flag_misinfo`: Dangerous misinformation or factually false claims.

### Observation Space (`SocialGuardObservation`)
- `posts`: A list of remaining `Post` objects (`text`, `reputation`, `post_id`).
- `current_post_index`: Pointer to the active item.
- `total_posts`: Length of the session.
- `reward`: Current step reward.
- `done`: Episode termination flag.

### Reward & Grading Function
- **Label Accuracy (70%)**: Correct classification adds 0.7 pts to the post reward.
- **Reasoning Bonus (30%)**: Rule-based (+0.3 pts) if reasoning is > 25 characters and includes policy keywords.
- **Safety & Precision**: Heavy penalties (-0.5 pts) for false positives on high-reputation users or missing high-threat violations.
- **Normalized Scores**: All final mission scores are normalized in the [0, 1] range to ensure cross-agent comparability.

---

## 🎯 Missions & Tasks

| Mission ID | Name | Difficulty | Description |
| :--- | :--- | :--- | :--- |
| `easy` | Easy Spam Detection | **Easy** | Catching obvious phishing from low-reputation bots. |
| `medium` | Medium Rep Moderation | **Medium** | Handling sarcasm and ambiguous misinformation. |
| `hard` | Hard Global Moderation | **Hard** | Detecting high-reputation toxicity and subtle hate speech. |

---

| Mission | Baseline Score (Qwen2.5-72B) | Difficulty Range (Judge Target) |
| :--- | :--- | :--- |
| **Easy Spam Detection** | **0.93** | 0.90 – 1.00 |
| **Medium Rep Moderation** | **0.78** | 0.65 – 0.85 |
| **Hard Global Moderation** | **0.70** (Capped) | 0.40 – 0.70 |

---

## ✅ Verified Performance (10-Sample Benchmark)

To ensure high-fidelity moderation, we recently verified the environment against an expanded 10-sample dataset.

| Metric | Result |
| :--- | :--- |
| **Model** | `Qwen/Qwen2.5-72B-Instruct` |
| **Accuracy** | **100.0%** |
| **Total Reward** | **+2.13** |
| **Dataset Source** | `data/training_samples.json` |

---

## 🛠️ Setup & Usage

### 🚀 Running the Environment (Docker)
The environment runs as a FastAPI server on port 7860.
```bash
docker build -t ai-social-guard .
docker run -p 7860:7860 ai-social-guard
```

### 🧠 Running Evaluation
`inference.py` interacts with the running container via REST API.

1. **Verify API availability**:
   ```bash
   curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "easy"}'
   ```

2. **Run the full mission suite**:
   ```bash
   python inference.py
   ```

---

## 🏷️ Tags
`openenv`, `content-moderation`, `safety`, `agent-benchmarking`, `hackathon`
