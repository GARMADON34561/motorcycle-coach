# server/tasks.py

from typing import List, Dict, Any, Callable
from models import MotorcycleAction

def grader_task1(action: MotorcycleAction) -> float:
    """Task 1: Lean left to avoid obstacle on right."""
    if action.steering < 0 and action.throttle > 0.2:
        return 1.0
    return 0.0

def grader_task2(action: MotorcycleAction) -> float:
    """Task 2: Brake hard when hazard is close."""
    if action.brake > 0.7 and action.throttle < 0.1:
        return 1.0
    return 0.0

def grader_task3(action: MotorcycleAction) -> float:
    """Task 3: Maintain speed while leaning slightly right."""
    if 0.4 < action.throttle < 0.8 and 0.0 < action.steering < 0.5:
        return 1.0
    return 0.0

ALL_TASKS: List[Dict[str, Any]] = [
    {
        "name": "obstacle_avoidance",
        "description": "Steer left to avoid obstacle on right.",
        "speed": 60.0,
        "lean": 0.0,
        "hazard_distance": 15.0,
        "hazard_type": "static",
        "grader": grader_task1
    },
    {
        "name": "emergency_braking",
        "description": "Brake hard when obstacle is very close.",
        "speed": 80.0,
        "lean": 0.0,
        "hazard_distance": 5.0,
        "hazard_type": "sudden",
        "grader": grader_task2
    },
    {
        "name": "cornering",
        "description": "Lean right while maintaining moderate throttle.",
        "speed": 50.0,
        "lean": 5.0,
        "hazard_distance": 100.0,
        "hazard_type": "none",
        "grader": grader_task3
    }
]