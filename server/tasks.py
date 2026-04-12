# server/tasks.py

from typing import List, Dict, Any, Callable
from models import MotorcycleAction

def safe_reward1(action: MotorcycleAction) -> float:
    steer = max(-1.0, min(1.0, action.steering))
    throttle = max(0.0, min(1.0, action.throttle))
    raw = 0.5 + 0.1 * steer + 0.1 * throttle
    return max(0.2, min(0.8, raw))

def safe_reward2(action: MotorcycleAction) -> float:
    brake = max(0.0, min(1.0, action.brake))
    throttle = max(0.0, min(1.0, action.throttle))
    raw = 0.6 - 0.2 * throttle + 0.2 * brake
    return max(0.2, min(0.8, raw))

def safe_reward3(action: MotorcycleAction) -> float:
    throttle = max(0.0, min(1.0, action.throttle))
    steer = max(-1.0, min(1.0, action.steering))
    raw = 0.4 + 0.2 * throttle + 0.1 * steer
    return max(0.2, min(0.8, raw))

ALL_TASKS: List[Dict[str, Any]] = [
    {
        "name": "obstacle_avoidance",
        "description": "Steer left to avoid obstacle on right.",
        "speed": 60.0,
        "lean": 0.0,
        "hazard_distance": 15.0,
        "hazard_type": "static",
        "reward_function": safe_reward1,   # <-- CHANGED KEY
        "min_score": 0.2,
        "max_score": 0.8,
    },
    {
        "name": "emergency_braking",
        "description": "Brake hard when obstacle is very close.",
        "speed": 80.0,
        "lean": 0.0,
        "hazard_distance": 5.0,
        "hazard_type": "sudden",
        "reward_function": safe_reward2,   # <-- CHANGED KEY
        "min_score": 0.2,
        "max_score": 0.8,
    },
    {
        "name": "cornering",
        "description": "Lean right while maintaining moderate throttle.",
        "speed": 50.0,
        "lean": 5.0,
        "hazard_distance": 100.0,
        "hazard_type": "none",
        "reward_function": safe_reward3,   # <-- CHANGED KEY
        "min_score": 0.2,
        "max_score": 0.8,
    },
]
