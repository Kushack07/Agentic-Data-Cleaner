import os
import json
from openai import OpenAI
from env.environment import DataPrepEnv

def run_inference():
    print("[START] Initializing DataPrepEnv Inference Baseline")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o")
    
    if not api_key:
        print("Warning: OPENAI_API_KEY is not set. Inference might fail if the client requires it.")
        
    client = OpenAI(api_key=api_key, base_url=base_url)
    
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
    
    while not done:
        print(f"[STEP] Current Step: {env.step_count}")
        
        # We pass the observation as json string
        obs_dict = obs.model_dump()
        messages.append({"role": "user", "content": f"Observation: {json.dumps(obs_dict)}"})
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            action_json_str = response.choices[0].message.content
            action_dict = json.loads(action_json_str)
            
            print(f"[STEP] Chosen action: {json.dumps(action_dict)}")
            
            # Step the environment
            obs, reward, done, info = env.step(action_dict)
            
            print(f"[STEP] Reward: {reward.value} | Info: {reward.reason}")
            
            # Add assistant's action to history
            messages.append({"role": "assistant", "content": action_json_str})
            # Provide reward feedback in the next turn
            messages.append({"role": "user", "content": f"Reward obtained: {reward.value}. Reason: {reward.reason}. Done: {done}"})
            
        except Exception as e:
            print(f"[STEP] Error during inference loop: {e}")
            break
            
    print("[END] Inference Complete.")
    print(f"Final Data shape: {env.current_df.shape}")

if __name__ == "__main__":
    run_inference()
