from typing import List, Dict, Any, Callable
from models import MotorcycleAction

def grader_task1(action: MotorcycleAction) -> float:
    steering_score = max(0.0, min(1.0, -action.steering))
    throttle_score = max(0.0, min(1.0, action.throttle))
    raw_reward = 0.4 * steering_score + 0.3 * throttle_score + 0.3
    return round(max(0.01, min(0.99, raw_reward)), 4)

def grader_task2(action: MotorcycleAction) -> float:
    brake_score = max(0.0, min(1.0, action.brake))
    throttle_penalty = 1.0 - max(0.0, min(1.0, action.throttle))
    raw_reward = 0.6 * brake_score + 0.3 * throttle_penalty + 0.1
    return round(max(0.01, min(0.99, raw_reward)), 4)

def grader_task3(action: MotorcycleAction) -> float:
    throttle_score = 1.0 - abs(action.throttle - 0.6)
    steering_score = 1.0 - abs(action.steering - 0.2)
    raw_reward = 0.5 * throttle_score + 0.4 * steering_score + 0.1
    return round(max(0.01, min(0.99, raw_reward)), 4)

ALL_TASKS: List[Dict[str, Any]] = [
    {
        "name": "obstacle_avoidance",
        "description": "Steer left to avoid obstacle on right.",
        "speed": 60.0,
        "lean": 0.0,
        "hazard_distance": 15.0,
        "hazard_type": "static",
        "grader": grader_task1,
        "min_score": 0.01,
        "max_score": 0.99,
    },
    {
        "name": "emergency_braking",
        "description": "Brake hard when obstacle is very close.",
        "speed": 80.0,
        "lean": 0.0,
        "hazard_distance": 5.0,
        "hazard_type": "sudden",
        "grader": grader_task2,
        "min_score": 0.01,
        "max_score": 0.99,
    },
    {
        "name": "cornering",
        "description": "Lean right while maintaining moderate throttle.",
        "speed": 50.0,
        "lean": 5.0,
        "hazard_distance": 100.0,
        "hazard_type": "none",
        "grader": grader_task3,
        "min_score": 0.01,
        "max_score": 0.99,
    },
]
