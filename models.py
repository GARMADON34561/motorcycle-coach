from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State

class MotorcycleAction(Action):
    throttle: float = Field(ge=0, le=1)
    brake: float = Field(ge=0, le=1)
    lean_angle: float = Field(ge=-45, le=45)
    steering: float = Field(ge=-1, le=1)

class MotorcycleObservation(Observation):
    task_id: int
    done: bool = False
    reward: float = 0.0

class MotorcycleState(State):
    current_task: int = 0
    total_reward: float = 0.0
