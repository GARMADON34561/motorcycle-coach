from uuid import uuid4
from typing import Optional, Dict, Any
from openenv.core.env_server.interfaces import Environment
from models import MotorcycleAction, MotorcycleObservation, MotorcycleState
from .tasks import ALL_TASKS

class MotorcycleEnvironment(Environment[MotorcycleAction, MotorcycleObservation, MotorcycleState]):
    def __init__(self):
        self._state = MotorcycleState(episode_id=str(uuid4()), step_count=0, current_task_index=0, total_reward=0.0)
        self.tasks = ALL_TASKS
        # Store last action for grading (optional)
        self._last_action = None

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs):
        self._state = MotorcycleState(episode_id=episode_id or str(uuid4()), step_count=0, current_task_index=0, total_reward=0.0)
        self._last_action = None
        return self._get_observation()

    def step(self, action: MotorcycleAction, **kwargs):
        self._last_action = action
        task_idx = self._state.current_task_index
        # Compute reward using the appropriate grader
        reward = self._grade_current_task(action)
        self._state.total_reward += reward
        self._state.step_count += 1
        self._state.current_task_index += 1
        done = self._state.current_task_index >= len(self.tasks)
        return self._get_observation(reward=reward, done=done), reward, done, {}

    def _grade_current_task(self, action: MotorcycleAction) -> float:
        """Internal grader for current task."""
        task = self.tasks[self._state.current_task_index]
        opt = task["optimal_action"]
        throttle_diff = abs(action.throttle - opt["throttle"])
        brake_diff = abs(action.brake - opt["brake"])
        lean_diff = abs(action.lean_angle - opt["lean_angle"]) / 30.0
        steer_diff = abs(action.steering - opt["steering"])
        score = 1.0 - (0.4 * throttle_diff + 0.3 * brake_diff + 0.2 * min(lean_diff, 1.0) + 0.1 * min(steer_diff, 1.0))
        return max(0.0, min(1.0, score))

    # --- Explicit grader methods for validator ---
    def grade_task1(self, action: MotorcycleAction) -> float:
        """Grader for Task 1 (Easy – gravel on straight)."""
        task = self.tasks[0]
        opt = task["optimal_action"]
        diff = (abs(action.throttle - opt["throttle"]) +
                abs(action.brake - opt["brake"]) +
                abs(action.lean_angle - opt["lean_angle"]) / 30.0 +
                abs(action.steering - opt["steering"]))
        return max(0.0, 1.0 - diff)

    def grade_task2(self, action: MotorcycleAction) -> float:
        """Grader for Task 2 (Medium – oil in turn)."""
        task = self.tasks[1]
        opt = task["optimal_action"]
        diff = (abs(action.throttle - opt["throttle"]) +
                abs(action.brake - opt["brake"]) +
                abs(action.lean_angle - opt["lean_angle"]) / 30.0 +
                abs(action.steering - opt["steering"]))
        return max(0.0, 1.0 - diff)

    def grade_task3(self, action: MotorcycleAction) -> float:
        """Grader for Task 3 (Hard – car cuts in)."""
        task = self.tasks[2]
        opt = task["optimal_action"]
        diff = (abs(action.throttle - opt["throttle"]) +
                abs(action.brake - opt["brake"]) +
                abs(action.lean_angle - opt["lean_angle"]) / 30.0 +
                abs(action.steering - opt["steering"]))
        return max(0.0, 1.0 - diff)

    def _get_observation(self, reward=0.0, done=False):
        if self._state.current_task_index >= len(self.tasks):
            return MotorcycleObservation(
                speed=0, lean_angle=0, hazard_distance=0, hazard_type="none",
                time_to_collision=0, done=done, reward=self._state.total_reward / len(self.tasks)
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