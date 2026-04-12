# server/motorcycle_environment.py

from uuid import uuid4
from typing import Optional, List, Dict, Any

from openenv.core.env_server.interfaces import Environment
from models import MotorcycleAction, MotorcycleObservation, MotorcycleState
from .tasks import ALL_TASKS

class MotorcycleEnvironment(Environment[MotorcycleAction, MotorcycleObservation, MotorcycleState]):
    tasks = ALL_TASKS

    def __init__(self):
        self._state = MotorcycleState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task_index=0,
            total_reward=0.0
        )

    def get_tasks(self) -> List[Dict[str, Any]]:
        return self.tasks

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs):
        self._state = MotorcycleState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_task_index=0,
            total_reward=0.0
        )
        return self._get_observation()

    def step(self, action: MotorcycleAction, **kwargs):
        task = self.tasks[self._state.current_task_index]
        reward = task["reward_function"](action)   # <-- CHANGED KEY

        self._state.total_reward += reward
        self._state.step_count += 1
        self._state.current_task_index += 1

        done = self._state.current_task_index >= len(self.tasks)
        return self._get_observation(reward=reward, done=done), reward, done, {}

    def _get_observation(self, reward=0.0, done=False):
        if self._state.current_task_index >= len(self.tasks):
            return MotorcycleObservation(
                speed=0,
                lean_angle=0,
                hazard_distance=0,
                hazard_type="none",
                time_to_collision=0,
                done=done,
                reward=self._state.total_reward / len(self.tasks)
            )

        task = self.tasks[self._state.current_task_index]
        return MotorcycleObservation(
            speed=task["speed"],
            lean_angle=task["lean"],
            hazard_distance=task["hazard_distance"],
            hazard_type=task["hazard_type"],
            time_to_collision=task["hazard_distance"] / max(task["speed"], 1.0),
            done=done,
            reward=reward
        )

    @property
    def state(self):
        return self._state
