from openenv.core.env_server.task import Task
from openenv.core.env_server.grader import Grader


class CorneringGrader(Grader):
    name = "smooth_cornering_grader"

    def grade(self, trajectory, **kwargs):
        return trajectory.final_reward


cornering_task = Task(
    id="smooth_cornering",
    description="Maintain optimal lean angle during a turn",
    graders=[CorneringGrader()],
)


class BrakingGrader(Grader):
    name = "emergency_braking_grader"

    def grade(self, trajectory, **kwargs):
        return trajectory.final_reward


braking_task = Task(
    id="emergency_braking",
    description="Brake safely and maintain control",
    graders=[BrakingGrader()],
)


class EfficiencyGrader(Grader):
    name = "fuel_safety_grader"

    def grade(self, trajectory, **kwargs):
        return trajectory.final_reward


efficiency_task = Task(
    id="fuel_safety",
    description="Balance throttle and braking for safety and fuel",
    graders=[EfficiencyGrader()],
)


ALL_TASKS = [
    cornering_task,
    braking_task,
    efficiency_task,
]