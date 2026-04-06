import os
import json
from openai import OpenAI
from env.environment import DataPrepEnv

def log_start(task: str, env: str, model: str):
    print(f"[START] {json.dumps({'task': task, 'env': env, 'model': model})}")

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    print(f"[STEP] {json.dumps({'step': step, 'action': action, 'reward': reward, 'done': done, 'error': error})}")

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] {json.dumps({'success': success, 'steps': steps, 'score': score, 'rewards': rewards})}")

API_BASE_URL = os.getenv("API_BASE_URL", "<your-api-base-url>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")



def run_inference():

    
    if not HF_TOKEN:
        print("Warning: HF_TOKEN is not set. Inference might fail if the client requires it.")
        
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    
    # We will test on a task config. (Hard by default for demonstrating capability)
    env = DataPrepEnv({"module": "tasks.task_hard", "max_steps": 10})
    obs = env.reset()
    
    done = False
    
    system_prompt = """You are an AI Data Scientist. Your task is to clean a tabular dataset.
You will be provided with a JSON observation containing the dataset head/tail preview, missing value counts, duplicate counts, column types, and summary statistics.
You must return a JSON dict mapping to the Action schema which has:
{
  "action_type": "fill_missing" | "drop_duplicates" | "remove_outliers" | "normalize_column" | "convert_data_type" | "submit",
  "column": "target_column_name" (null for drop_duplicates/submit),
  "parameters": {} (e.g. {"fill_value": "median"} or {"target_type": "float"}).
}
Always end by executing the "submit" action when the dataset is clean.
"""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Track rewards for final score
    rewards_history = []
    
    log_start(task="DataPrep-Hard", env="DataPrepEnv", model=MODEL_NAME)
    
    while not done:
        
        # We pass the observation as json string
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
        messages.append({"role": "user", "content": f"Observation: {json.dumps(obs_dict)}"})
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            action_json_str = response.choices[0].message.content
            action_dict = json.loads(action_json_str)
            
            # Step the environment
            obs, reward_obj, done, info = env.step(action_dict)
            
            # Log the step strictly
            # Since action_dict is a dict, we dump it to str
            rewards_history.append(reward_obj.value if hasattr(reward_obj, "value") else getattr(reward_obj, "reward", 0.0))
            
            log_step(
                step=env.step_count, 
                action=json.dumps(action_dict), 
                reward=rewards_history[-1], 
                done=done, 
                error=None
            )

            
            # Add assistant's action to history
            messages.append({"role": "assistant", "content": action_json_str})
            messages.append({"role": "user", "content": f"Reward obtained: {rewards_history[-1]}. Done: {done}"})
            
        except Exception as e:
            log_step(step=env.step_count, action="ERROR", reward=0.0, done=True, error=str(e))
            break
            
    # Calculate final score and log end
    score = sum(rewards_history) / len(rewards_history) if rewards_history else 0.0
    success = score > 0.5
    log_end(success=success, steps=env.step_count, score=score, rewards=rewards_history)


if __name__ == "__main__":
    run_inference()
