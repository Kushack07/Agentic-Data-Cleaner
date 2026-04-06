from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from env.environment import DataPrepEnv as LegacyDataPrepEnv
from models import DataPrepAction, DataPrepObservation


class DataPrepEnvironment(Environment):
    def __init__(self, task_module: str = "tasks.task_hard", max_steps: int = 20):
        self.task_module = task_module
        self.max_steps = max_steps
        self._env = LegacyDataPrepEnv(
            {"module": self.task_module, "max_steps": self.max_steps}
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def _to_observation(
        self,
        observation,
        reward: float = 0.0,
        done: bool = False,
        info: dict | None = None,
    ) -> DataPrepObservation:
        payload = observation.model_dump() if hasattr(observation, "model_dump") else observation
        return DataPrepObservation(
            **payload,
            reward=reward,
            done=done,
            info=info or {},
        )

    def reset(self) -> DataPrepObservation:
        self._env = LegacyDataPrepEnv(
            {"module": self.task_module, "max_steps": self.max_steps}
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)
        observation = self._env.reset()
        return self._to_observation(
            observation,
            reward=0.0,
            done=False,
            info={"task_module": self.task_module},
        )

    def step(self, action: DataPrepAction) -> DataPrepObservation:
        observation, reward_obj, done, info = self._env.step(action.model_dump())
        self._state.step_count = self._env.step_count
        reward = reward_obj.value if hasattr(reward_obj, "value") else float(reward_obj)
        return self._to_observation(
            observation,
            reward=reward,
            done=done,
            info=info,
        )

    @property
    def state(self) -> State:
        return self._state
