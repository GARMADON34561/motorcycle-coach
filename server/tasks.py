from models import MotorcycleAction

def _grader_common(action: MotorcycleAction, optimal: dict) -> float:
    diff = (abs(action.throttle - optimal["throttle"]) +
            abs(action.brake - optimal["brake"]) +
            abs(action.lean_angle - optimal["lean_angle"]) / 30.0 +
            abs(action.steering - optimal["steering"]))
    return max(0.0, min(1.0, 1.0 - diff))

# Task 1 grader
def grader_1(action: MotorcycleAction) -> float:
    return _grader_common(action, {"throttle": 0.0, "brake": 0.4, "lean_angle": 0.0, "steering": 0.0})

# Task 2 grader
def grader_2(action: MotorcycleAction) -> float:
    return _grader_common(action, {"throttle": 0.2, "brake": 0.1, "lean_angle": 10.0, "steering": 0.5})

# Task 3 grader
def grader_3(action: MotorcycleAction) -> float:
    return _grader_common(action, {"throttle": 0.0, "brake": 0.8, "lean_angle": 0.0, "steering": -0.3})

ALL_TASKS = [
    {
        "name": "gravel_on_straight",
        "speed": 80.0, "lean": 0.0, "hazard_distance": 50.0, "hazard_type": "gravel",
        "optimal_action": {"throttle": 0.0, "brake": 0.4, "lean_angle": 0.0, "steering": 0.0},
        "grader": grader_1,
    },
    {
        "name": "oil_in_turn",
        "speed": 60.0, "lean": 15.0, "hazard_distance": 30.0, "hazard_type": "oil",
        "optimal_action": {"throttle": 0.2, "brake": 0.1, "lean_angle": 10.0, "steering": 0.5},
        "grader": grader_2,
    },
    {
        "name": "car_cuts_in",
        "speed": 90.0, "lean": 5.0, "hazard_distance": 25.0, "hazard_type": "car",
        "optimal_action": {"throttle": 0.0, "brake": 0.8, "lean_angle": 0.0, "steering": -0.3},
        "grader": grader_3,
    },
]