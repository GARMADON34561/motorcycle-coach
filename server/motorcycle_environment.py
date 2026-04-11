from uuid import uuid4
from typing import Optional
from openenv.core.env_server.interfaces import Environment
from models import MotorcycleAction, MotorcycleObservation, MotorcycleState

class MotorcycleEnvironment(Environment[MotorcycleAction, MotorcycleObservation, MotorcycleState]):
    def __init__(self):
        self._state = MotorcycleState(episode_id=str(uuid4()), step_count=0, current_task=0, total_reward=0.0)

    def reset(self, seed=None, episode_id=None, **kwargs):
        self._state = MotorcycleState(episode_id=episode_id or str(uuid4()), step_count=0, current_task=0, total_reward=0.0)
        return MotorcycleObservation(task_id=0, done=False, reward=0.0)

    def step(self, action: MotorcycleAction, **kwargs):
        task = self._state.current_task
        # 3 graders
        if task == 0:
            ideal_lean = 25.0
            error = abs(action.lean_angle - ideal_lean) / 45.0
            reward = 1.0 - min(1.0, error)
        elif task == 1:
            stopping = action.brake * 15 + abs(action.steering) * 5
            if stopping >= 20:
                reward = 0.01
            else:
                reward = 0.5 + (1.0 - stopping/20) * 0.49
        else:
            fuel = action.throttle * 0.5
            safety = 1.0 - action.brake
            reward = (1.0 - fuel/5) * 0.5 + safety * 0.5

        reward = max(0.01, min(0.99, reward))
        self._state.total_reward += reward
        self._state.current_task += 1
        self._state.step_count += 1
        done = self._state.current_task >= 3
        if done:
            final_reward = self._state.total_reward / 3.0
            final_reward = max(0.01, min(0.99, final_reward))
            obs = MotorcycleObservation(task_id=3, done=True, reward=final_reward)
        else:
            obs = MotorcycleObservation(task_id=self._state.current_task, done=False, reward=reward)
        return obs, reward, done, {}

    @property
    def state(self):
        return self._state
    
    @property
    def tasks(self):
        return [
            "smooth_cornering",
            "emergency_braking",
            "fuel_efficiency_and_safety"
        ]
