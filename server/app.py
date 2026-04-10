import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uuid

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

# ----- Environment -----
class MotorcycleEnvironment:
    def __init__(self):
        self.tasks = [
            {"type": "corner", "initial_speed": 40.0, "turn_radius": 15.0, "road_condition": "dry", "max_safe_lean": 35.0},
            {"type": "emergency", "initial_speed": 60.0, "obstacle_distance": 20.0, "road_condition": "wet"},
            {"type": "cruise", "distance_km": 5.0, "traffic_density": 0.5, "road_condition": "dry"}
        ]
        self.current_task = 0
        self.total_reward = 0.0
        self.step_count = 0

    def reset(self):
        self.current_task = 0
        self.total_reward = 0.0
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        task = self.tasks[self.current_task]
        if task["type"] == "corner":
            return MotorcycleObservation(
                speed_kmh=task["initial_speed"], lean_angle=0.0, distance_to_obstacle_m=999.0,
                fuel_level_l=5.0, road_condition=task["road_condition"], turn_radius_m=task["turn_radius"],
                headway_seconds=999.0, done=False, reward=0.0
            )
        elif task["type"] == "emergency":
            return MotorcycleObservation(
                speed_kmh=task["initial_speed"], lean_angle=0.0, distance_to_obstacle_m=task["obstacle_distance"],
                fuel_level_l=5.0, road_condition=task["road_condition"], turn_radius_m=999.0,
                headway_seconds=999.0, done=False, reward=0.0
            )
        else:
            return MotorcycleObservation(
                speed_kmh=50.0, lean_angle=0.0, distance_to_obstacle_m=999.0, fuel_level_l=10.0,
                road_condition=task["road_condition"], turn_radius_m=999.0, headway_seconds=2.0,
                done=False, reward=0.0
            )

    def step(self, action: MotorcycleAction):
        task = self.tasks[self.current_task]
        reward = 0.0
        if task["type"] == "corner":
            ideal_lean = min(35.0, (action.throttle * 40)**2 / (task["turn_radius"] * 9.8) * 10)
            lean_error = abs(action.lean_angle - ideal_lean) / task["max_safe_lean"]
            raw = 1.0 - min(1.0, lean_error)
            reward = max(0.01, min(0.99, raw))
        elif task["type"] == "emergency":
            stopping_distance = (action.brake * 15) + (abs(action.steering) * 5)
            if stopping_distance >= task["obstacle_distance"]:
                reward = 0.01
            else:
                smoothness = 1.0 - abs(action.steering) * 0.5
                raw = 0.5 + smoothness * 0.4
                reward = max(0.01, min(0.99, raw))
        else:
            fuel_used = action.throttle * 0.5
            headway_safety = min(1.0, max(0.0, (action.brake * 2 + 1) / 3))
            raw = (1 - fuel_used/5) * 0.5 + headway_safety * 0.5
            reward = max(0.01, min(0.99, raw))

        self.total_reward += reward
        self.current_task += 1
        done = self.current_task >= len(self.tasks)
        if done:
            final_reward = self.total_reward / len(self.tasks)
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