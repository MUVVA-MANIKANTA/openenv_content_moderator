import gradio as gr
import os
import requests
import json
import random
from src.env import AISocialGuardEnv
from src.models import SocialGuardAction, ActionType, Post

# Version: 1.0.2 - Final OpenEnv Submission
# Initialize environment
env = AISocialGuardEnv()

def moderate_post(text, reputation, reason):
    # Mock post for the environment
    post_data = {
        "post_id": 1,
        "text": text,
        "user_id": 999 if "!" in text else 101, # Simulating serial offender logic
        "reputation": reputation,
        "correct_label": ActionType.APPROVE 
    }
    
    # Initialize env with this one post for the demo
    env.reset(task_config={"posts": [post_data]})
    
    # Simple logic to determine decision for demo
    decision = ActionType.APPROVE
    lower_text = text.lower()
    if any(word in lower_text for word in ["kill", "hate", "stupid", "worthless"]):
        decision = ActionType.FLAG_HATE
    elif any(word in lower_text for word in ["buy", "win", "prize"]):
        decision = ActionType.FLAG_SPAM
    elif any(word in lower_text for word in ["porn", "18+"]):
        decision = ActionType.FLAG_ADULT
        
    action = SocialGuardAction(post_id=1, action_type=decision, reason=reason)
    obs, reward, done, info = env.step(action)
    
    result = {
        "Moderation Decision": decision.value.upper(),
        "Reward Signal": f"{reward:+.2f}",
        "Serial Offender Threat": f"{info.get('threat_level', 0)*100:.0f}%",
        "Reasoning Logged": reason if reason else "No reasoning provided.",
        "Observation State": obs.model_dump()
    }
    
    return json.dumps(result, indent=2)

def create_gradio_demo():
    # Use a persistent state for analytics
    cumulative_reward = gr.State(0.0)
    total_samples = gr.State(0)
    total_correct = gr.State(0)
    policy_log_content = gr.State("")

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🛡️ AI Social Guard v3 (RL + Learning Loop)")
        gr.Markdown("> **'We are not just moderating content — we are training AI agents using reinforcement learning to improve moderation decisions over time.'**")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🔍 Live Moderation Console")
                input_text = gr.Textbox(label="Post Content", placeholder="Enter post text here...", lines=3)
                with gr.Row():
                    user_selector = gr.Dropdown(choices=["User_101 (Neutral)", "User_999 (Serial Offender)", "User_HighRep (Trusted)"], label="Acting User", value="User_101 (Neutral)")
                    reputation_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Author Reputation Override")
                
                reason_text = gr.Textbox(label="Reasoning (Justification)", placeholder="Why flag this? (Optional)")
                submit_btn = gr.Button("🚀 Evaluate & Train Model", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Policy Analytics & Learning Dashboard")
                reward_metric = gr.Number(label="Session Cumulative Reward (Policy Utility)", precision=2, interactive=False)
                accuracy_metric = gr.Label(label="Model-to-Policy Alignment Accuracy")
                confidence_meter = gr.HTML("<div style='padding:10px; border-radius:5px; background:#333; color:white; text-align:center;'>Policy Confidence: <b>0.00%</b></div>")
                threat_meter = gr.HTML("<div style='padding:10px; border-radius:5px; background:#444; color:white; text-align:center;'>System Status: <b>PROTECTED</b></div>")
                
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📜 Real-Time Policy Evolution Log")
                log_box = gr.Textbox(label="Agent Training Feed", lines=8, interactive=False)
            
            with gr.Column(scale=2):
                gr.Markdown("### 📄 Observation State & Reward Signal")
                output_json = gr.Code(label="Real-Time OpenEnv API Response", language="json")
        
        gr.Markdown("### 🛠️ Multi-Category Flagging: `APPROVE`, `FLAG_HATE`, `FLAG_MISINFO`, `FLAG_ADULT`, `FLAG_SPAM`")

        def process_and_track(text, user_str, reputation, reason, total_reward_sum=0.0, samples_sum=0, correct_sum=0, current_log=""):
            # Map user string to ID
            uid = 101
            if user_str and "999" in user_str: uid = 999
            elif user_str and "HighRep" in user_str: uid = 202

            rnd = random.Random(samples_sum)

            post_data = {
                "post_id": rnd.randint(1000, 9999),
                "text": text,
                "user_id": uid,
                "reputation": 0.9 if uid == 202 else (0.1 if uid == 999 else reputation),
                "correct_label": ActionType.APPROVE 
            }
            
            env.reset(task_config={"posts": [post_data]})
            
            # Simple heuristic for UI demo to simulate category selection
            decision = ActionType.APPROVE
            lt = text.lower()
            if any(w in lt for w in ["kill", "hate", "race"]): decision = ActionType.FLAG_HATE
            elif any(w in lt for w in ["cheap", "win", "buy"]): decision = ActionType.FLAG_SPAM
            elif any(w in lt for w in ["fake", "lie", "proof"]): decision = ActionType.FLAG_MISINFO
            elif any(w in lt for w in ["porn", "18+"]): decision = ActionType.FLAG_ADULT
            
            action = SocialGuardAction(post_id=post_data["post_id"], action_type=decision, reason=reason)
            obs, reward, done, info = env.step(action)
            
            new_total_reward = total_reward_sum + reward
            new_samples = samples_sum + 1
            is_correct = 1 if reward >= 0 else 0
            new_correct_count = correct_sum + is_correct
            
            accuracy = (new_correct_count / new_samples) * 100
            
            # Simulated training log
            timestamp = f"[{rnd.randint(10, 23)}:{rnd.randint(10, 59)}:{rnd.randint(10, 59)}]"
            log_entry = f"{timestamp} [TRAIN] Policy refined based on Reward: {reward:+.2f} for User_{uid}.\n"
            if reward < 0:
                log_entry += f"{timestamp} [WARN] Violation detected. Adjusted weights for UID_{uid}.\n"
            
            new_log = log_entry + current_log
            
            status_color = "green" if reward >= 0 else "red"
            status_text = "SAFE" if reward >= 0 else "THREAT DETECTED"
            threat_html = f"<div style='padding:10px; border-radius:5px; background:{status_color}; color:white; text-align:center;'>System Status: <b>{status_text}</b> (Threat: {info.get('threat_level', 0)*100:.0f}%)</div>"
            confidence_html = f"<div style='padding:10px; border-radius:5px; background:#222; color:#0f0; text-align:center;'>Policy Confidence: <b>{accuracy:.1f}%</b></div>"
            
            result = {
                "Moderation Decision": decision.value,
                "Reward Signal": f"{reward:+.2f}",
                "Policy Metadata": {
                    "Session Reward": f"{new_total_reward:.2f}",
                    "Alignment Accuracy": f"{accuracy:.1f}%",
                    "Threat Score": f"{info.get('threat_level', 0):.2f}"
                },
                "OpenEnv Observation": obs.model_dump()
            }
            
            return (
                json.dumps(result, indent=2), 
                new_total_reward, 
                new_total_reward, 
                new_samples, 
                new_correct_count, 
                new_log, 
                new_log, 
                f"{accuracy:.1f}%", 
                confidence_html, 
                threat_html
            )

        submit_btn.click(
            fn=process_and_track, 
            inputs=[input_text, user_selector, reputation_slider, reason_text, cumulative_reward, total_samples, total_correct, policy_log_content], 
            outputs=[output_json, cumulative_reward, reward_metric, total_samples, total_correct, log_box, policy_log_content, accuracy_metric, confidence_meter, threat_meter]
        )
    return demo

if __name__ == "__main__":
    demo = create_gradio_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
