import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Tuple

# ----- Models -----
class MotorcycleAction(BaseModel):
    throttle: float = Field(ge=0, le=1)
    brake: float = Field(ge=0, le=1)
    lean_angle: float = Field(ge=-45, le=45)
    steering: float = Field(ge=-1, le=1)

class MotorcycleObservation(BaseModel):
    speed_kmh: float
    lean_angle: float
    distance_to_obstacle_m: float
    fuel_level_l: float
    road_condition: str
    turn_radius_m: float
    headway_seconds: float
    done: bool = False
    reward: float = 0.0

# ----- Environment with three explicit graders -----
class MotorcycleEnvironment:
    def __init__(self):
        self.task_index = 0  # 0,1,2
        self.total_reward = 0.0

    def reset(self):
        self.task_index = 0
        self.total_reward = 0.0
        return self._get_obs()

    def _get_obs(self):
        # Return observation based on current task
        if self.task_index == 0:  # cornering
            return MotorcycleObservation(
                speed_kmh=40.0, lean_angle=0.0, distance_to_obstacle_m=999.0,
                fuel_level_l=5.0, road_condition="dry", turn_radius_m=15.0,
                headway_seconds=999.0, done=False, reward=0.0
            )
        elif self.task_index == 1:  # emergency
            return MotorcycleObservation(
                speed_kmh=60.0, lean_angle=0.0, distance_to_obstacle_m=20.0,
                fuel_level_l=5.0, road_condition="wet", turn_radius_m=999.0,
                headway_seconds=999.0, done=False, reward=0.0
            )
        else:  # cruise
            return MotorcycleObservation(
                speed_kmh=50.0, lean_angle=0.0, distance_to_obstacle_m=999.0,
                fuel_level_l=10.0, road_condition="dry", turn_radius_m=999.0,
                headway_seconds=2.0, done=False, reward=0.0
            )

    def step(self, action: MotorcycleAction):
        # Grade the current task and move to next
        if self.task_index == 0:
            reward = self._grader_cornering(action)
        elif self.task_index == 1:
            reward = self._grader_emergency(action)
        else:
            reward = self._grader_cruise(action)

        # Clamp reward strictly between 0 and 1
        reward = max(0.01, min(0.99, reward))
        self.total_reward += reward
        self.task_index += 1
        done = self.task_index >= 3

        if done:
            final_reward = self.total_reward / 3.0
            final_reward = max(0.01, min(0.99, final_reward))
            obs = MotorcycleObservation(
                speed_kmh=0, lean_angle=0, distance_to_obstacle_m=0, fuel_level_l=0,
                road_condition="", turn_radius_m=0, headway_seconds=0, done=True, reward=final_reward
            )
        else:
            obs = self._get_obs()
            obs.reward = reward
            obs.done = False
        return obs

    def _grader_cornering(self, action: MotorcycleAction) -> float:
        # Ideal lean angle for 40 km/h, radius 15m
        ideal_lean = min(35.0, (action.throttle * 40)**2 / (15.0 * 9.8) * 10)
        lean_error = abs(action.lean_angle - ideal_lean) / 35.0
        score = 1.0 - min(1.0, lean_error)
        return max(0.01, min(0.99, score))

    def _grader_emergency(self, action: MotorcycleAction) -> float:
        stopping_distance = (action.brake * 15) + (abs(action.steering) * 5)
        if stopping_distance >= 20.0:
            return 0.01  # crash
        smoothness = 1.0 - abs(action.steering) * 0.5
        score = 0.5 + smoothness * 0.4
        return max(0.01, min(0.99, score))

    def _grader_cruise(self, action: MotorcycleAction) -> float:
        fuel_used = action.throttle * 0.5
        headway_safety = min(1.0, max(0.0, (action.brake * 2 + 1) / 3))
        score = (1 - fuel_used/5) * 0.5 + headway_safety * 0.5
        return max(0.01, min(0.99, score))

# ----- FastAPI -----
app = FastAPI()
env = MotorcycleEnvironment()

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs.dict(), "reward": obs.reward, "done": obs.done}

@app.post("/step")
def step(action: MotorcycleAction):
    obs = env.step(action)
    return {"observation": obs.dict(), "reward": obs.reward, "done": obs.done}

@app.get("/")
def root():
    return {"status": "ready"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()