from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State

class MotorcycleAction(Action):
    throttle: float = Field(ge=0.0, le=1.0, description="Throttle 0..1")
    brake: float = Field(ge=0.0, le=1.0, description="Brake 0..1")
    lean_angle: float = Field(description="Lean angle in degrees")
    steering: float = Field(ge=-1.0, le=1.0, description="Steering input -1..1")

class MotorcycleObservation(Observation):
    speed: float = Field(description="Current speed (km/h)")
    lean_angle: float = Field(description="Current lean angle (deg)")
    hazard_distance: float = Field(description="Distance to next hazard (m)")
    hazard_type: str = Field(description="hazard type: gravel, oil, pothole, car")
    time_to_collision: float = Field(description="Seconds until impact if no action")
    done: bool = False
    reward: float = 0.0

class MotorcycleState(State):
    current_task_index: int = 0
    total_reward: float = 0.0