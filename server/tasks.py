TASKS = [
    # Easy: simple hazard – gravel on straight road
    {
        "speed": 80.0,
        "lean": 0.0,
        "hazard_distance": 50.0,
        "hazard_type": "gravel",
        "optimal_action": {"throttle": 0.0, "brake": 0.4, "lean_angle": 0.0, "steering": 0.0},
        "description": "Gravel patch ahead on straight road. Reduce speed smoothly."
    },
    # Medium: decreasing radius turn with oil
    {
        "speed": 60.0,
        "lean": 15.0,
        "hazard_distance": 30.0,
        "hazard_type": "oil",
        "optimal_action": {"throttle": 0.2, "brake": 0.1, "lean_angle": 10.0, "steering": 0.5},
        "description": "Oil spill in a right‑hand bend. Slight throttle, reduce lean, steer carefully."
    },
    # Hard: car pulls out unexpectedly
    {
        "speed": 90.0,
        "lean": 5.0,
        "hazard_distance": 25.0,
        "hazard_type": "car",
        "optimal_action": {"throttle": 0.0, "brake": 0.8, "lean_angle": 0.0, "steering": -0.3},
        "description": "Car cuts in from left. Emergency brake and slight swerve right."
    }
]

ALL_TASKS = TASKS  # for easier import