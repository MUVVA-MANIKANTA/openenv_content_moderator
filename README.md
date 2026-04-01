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

*   **Multi-Category Moderation**: Automated classification into **HATE, MISINFO, VANDALISM, ADULT, SPAM, and OFFENSIVE** categories.
*   **Serial Offender Tracking**: Proactive risk signaling for repeat violators (e.g., User 999).
*   **Dynamic Reputation Engine**: Context-aware rewards that vary based on the author's history.
*   **AIEvaluation (v2)**: Programmatic graders mapping to OpenEnv Task scoring (0.0 to 1.0).

> **"We are not just moderating content — we are training AI agents using reinforcement learning to improve moderation decisions over time."**

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

### Reward Function
- **Precision Bonus**: Rewards for correct categorization (+0.10 to +0.20 per flag).
- **Reputation Penalty**: Incorrectly flagging high-reputation users is penalized.
- **Serial Offender Penalty**: Failing to catch a repeat offender (User 999) results in escalating negative rewards.
- **Reasoning Bonus**: Tiny bonus (+0.01) for providing a `reason` > 5 characters.

---

## 🎯 Missions & Tasks

| Mission ID | Name | Difficulty | Description |
| :--- | :--- | :--- | :--- |
| `easy_spam` | Easy Spam Detection | **Easy** | Catching obvious phishing from low-reputation bots. |
| `medium_reputation` | Medium Rep Moderation | **Medium** | Handling a mix of spam and adult content. |
| `hard_global` | Hard Global Moderation | **Hard** | Detecting misinformation and subtle hate speech. |
| `crisis_response` | Crisis Response | **EXTREME** | High-velocity misinformation attack from a serial offender. |

---

## 📊 Baseline Performance (OpenAI GPT-4o)

| Mission | Baseline Score (Accuracy) | Avg. Reward |
| :--- | :--- | :--- |
| **Easy Spam Detection** | 1.00 | +0.15 |
| **Medium Rep Moderation** | 0.92 | +0.12 |
| **Hard Global Moderation** | 0.70 | +0.08 |
| **Crisis Response** | 0.65 | -0.10 |

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
Both `inference.py` and `test_samples.py` automatically load configuration from a `.env` file.

1. **Create a `.env` file** in the project root:
   ```bash
   HF_TOKEN=your_token_here
   MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
   API_BASE_URL=https://router.huggingface.co/v1
   ```

2. **Run the 10-sample verification**:
   ```bash
   python test_samples.py
   ```

3. **Run the full mission suite**:
   ```bash
   python inference.py
   ```

### 🖥️ Human-in-the-Loop UI
You can interact with the environment visually via the Gradio UI mounted at `/ui` (e.g., `http://localhost:7860/ui`).

---

## 🏷️ Tags
`openenv`, `content-moderation`, `safety`, `agent-benchmarking`, `hackathon`
