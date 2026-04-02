import importlib
import json
import traceback
from typing import Dict, Any, Tuple

from pydantic import ValidationError

from env.actions import Observation, Action, RewardSchema, apply_action
from env.reward import compute_reward

class DataPrepEnv:
    def __init__(self, task_config: Dict[str, Any] = None):
        """
        Initializes the environment based on task configuration.
        """
        self.task_config = task_config or {}
        self.task_module_name = self.task_config.get("module", "tasks.task_easy")
        
        # Load the task generator dynamically
        task_module = importlib.import_module(self.task_module_name)
        self.generate_dataset = getattr(task_module, "generate_dataset")
        
        self.max_steps = self.task_config.get("max_steps", 20)
        self.step_count = 0
        self.done = False
        
        # Internal state variables
        self.initial_df = None
        self.current_df = None

    def reset(self) -> Observation:
        """
        Resets the environment and returns the initial observation.
        """
        self.initial_df = self.generate_dataset()
        self.current_df = self.initial_df.copy()
        
        self.step_count = 0
        self.done = False
        
        return self.state()

    def state(self) -> Observation:
        """
        Returns the current state observation.
        """
        if self.current_df is None:
            raise ValueError("Environment must be reset before calling state()")
            
        df = self.current_df
        
        # Build dataset preview (head and tail combined into markdown string)
        if len(df) > 10:
            preview = df.head().to_markdown() + "\n...\n" + df.tail().to_markdown()
        else:
            preview = df.to_markdown()
            
        return Observation(
            dataset_preview=preview,
            missing_value_counts=df.isna().sum().to_dict(),
            duplicate_counts=int(df.duplicated().sum()),
            column_types={col: str(dtype) for col, dtype in df.dtypes.items()},
            summary_statistics=df.describe().to_dict()
        )

    def step(self, action: Dict[str, Any]) -> Tuple[Observation, RewardSchema, bool, Dict[str, Any]]:
        """
        Takes an action and updates the environment state.
        Returns observation, reward, done, info.
        """
        if self.done:
             raise RuntimeError("Environment is done. Please reset.")

        self.step_count += 1
        
        try:
            # Parse action using the Pydantic schema
            parsed_action = Action(**action)
        except ValidationError as e:
            # If the agent sends an invalid action structure
            obs = self.state()
            reward_val, info_msg = compute_reward("invalid", self.current_df, self.current_df, error_msg=str(e))
            self.done = True # Abort on schema violations to prevent infinite loops of bad schema
            return obs, RewardSchema(value=reward_val, reason=info_msg), self.done, {"error": str(e)}

        before_df = self.current_df.copy()
        after_df, error_msg = apply_action(before_df, parsed_action)
        
        # Update current state
        self.current_df = after_df

        # Compute dense reward
        reward_val, info_msg = compute_reward(parsed_action.action_type, before_df, after_df, error_msg)
        
        if parsed_action.action_type == "submit":
            self.done = True
            
        if self.step_count >= self.max_steps:
             self.done = True
             info_msg += f" Max steps ({self.max_steps}) reached."
             
        obs = self.state()
        reward = RewardSchema(value=reward_val, reason=info_msg)
        
        return obs, reward, self.done, {"action_taken": parsed_action.action_type, "info": info_msg}

    def get_final_dataset(self):
        """Used by the grader to evaluate the final state."""
        return self.current_df
