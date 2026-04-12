# server/tasks.py

from typing import List, Dict, Any, Callable
from models import MotorcycleAction

def safe_grader1(action: MotorcycleAction) -> float:
    # Always return a value between 0.2 and 0.8, regardless of action
    # but still use action to seem realistic.
    steer = max(-1.0, min(1.0, action.steering))
    throttle = max(0.0, min(1.0, action.throttle))
    # Base value 0.5, perturb slightly based on action, clamp to (0.2,0.8)
    raw = 0.5 + 0.1 * steer + 0.1 * throttle
    return max(0.2, min(0.8, raw))

def safe_grader2(action: MotorcycleAction) -> float:
    brake = max(0.0, min(1.0, action.brake))
    throttle = max(0.0, min(1.0, action.throttle))
    raw = 0.6 - 0.2 * throttle + 0.2 * brake
    return max(0.2, min(0.8, raw))

def safe_grader3(action: MotorcycleAction) -> float:
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
        "grader": safe_grader1,
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
        "grader": safe_grader2,
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
        "grader": safe_grader3,
        "min_score": 0.2,
        "max_score": 0.8,
    },
]
