import gradio as gr
import random
from .models import ActionType, SocialGuardAction

def create_gradio_demo():
    with gr.Blocks(title="AI Social Guard Demo") as demo:
        gr.Markdown("# AI Social Guard Moderation Interface")
        
        status = gr.Textbox(label="Status", interactive=False)
        current_post_out = gr.JSON(label="Current Post")
        
        with gr.Row():
            btn_approve = gr.Button("Approve")
            btn_spam = gr.Button("Flag Spam")
            btn_hate = gr.Button("Flag Hate")
            btn_adult = gr.Button("Flag Adult")
            btn_misinfo = gr.Button("Flag Misinfo")
            
        history = gr.JSON(label="Moderation History")
            
        def take_action(action_type_str):
            from .server import env
            state = env.state()
            idx = state.current_index
            posts = state.all_posts
            
            if idx >= len(posts):
                return "Done. No more posts.", None, [h for h in state.history]
                
            post_id = posts[idx].post_id
            action = SocialGuardAction(
                post_id=post_id,
                action_type=ActionType(action_type_str),
                reason="Human moderation from UI"
            )
            
            obs, reward, done, info = env.step(action)
            reward_val = reward.value if hasattr(reward, 'value') else float(reward)
            
            samples_sum = post_id + sum(ord(c) for c in action_type_str)
            rnd = random.Random(samples_sum)
            timestamp = f"[{rnd.randint(10,23)}:{rnd.randint(10,59)}:{rnd.randint(10,59)}]"
            
            msg = f"{timestamp} Action: {action_type_str} | Reward: {reward_val:+.2f} | Reason: {reward.reason if hasattr(reward, 'reason') else ''}"
            
            if reward_val >= 0.5:
                msg += " (Good job!)"
                
            new_total_reward = sum([h["reward"]["value"] for h in env.state().history]) + reward_val
            
            return msg, obs.current_post, [h for h in env.state().history]
            
        btn_approve.click(fn=lambda: take_action(ActionType.APPROVE.value), outputs=[status, current_post_out, history])
        btn_spam.click(fn=lambda: take_action(ActionType.FLAG_SPAM.value), outputs=[status, current_post_out, history])
        btn_hate.click(fn=lambda: take_action(ActionType.FLAG_HATE.value), outputs=[status, current_post_out, history])
        btn_adult.click(fn=lambda: take_action(ActionType.FLAG_ADULT.value), outputs=[status, current_post_out, history])
        btn_misinfo.click(fn=lambda: take_action(ActionType.FLAG_MISINFO.value), outputs=[status, current_post_out, history])
        
    return demo
