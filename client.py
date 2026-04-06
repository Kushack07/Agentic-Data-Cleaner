from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from models import DataPrepAction, DataPrepObservation


class DataPrepEnvClient(EnvClient[DataPrepAction, DataPrepObservation, State]):
    def _step_payload(self, action: DataPrepAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[DataPrepObservation]:
        observation_payload = dict(payload.get("observation", {}))
        observation = DataPrepObservation(
            **observation_payload,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
