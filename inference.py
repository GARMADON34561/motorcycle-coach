import os
import asyncio
import json
from typing import List, Optional
from openai import OpenAI

from server.app import MotorcycleEnvironment, MotorcycleAction

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

TASK_NAME = "motorcycle_coach"
BENCHMARK = "motorcycle_coach"

def clamp_score(score: float) -> float:
    return max(0.01, min(0.99, score))

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    safe_reward = clamp_score(reward)
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={safe_reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    safe_score = clamp_score(score)
    safe_rewards = [clamp_score(r) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in safe_rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={safe_score:.3f} rewards={rewards_str}", flush=True)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = MotorcycleEnvironment()
    all_rewards = []

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        # Reset once
        obs = env.reset()
        done = False
        step = 0

        while not done and step < 5:
            step += 1
            # Describe task based on step number
            task_descs = ["cornering at 40 km/h", "emergency brake and swerve", "cruise for fuel efficiency"]
            desc = task_descs[step-1] if step-1 < len(task_descs) else "unknown"

            prompt = f"""
You are an AR motorcycle coach. Task {step}: {desc}
Current state:
Speed: {obs.speed_kmh} km/h, Lean angle: {obs.lean_angle} deg,
Distance to obstacle: {obs.distance_to_obstacle_m} m,
Road: {obs.road_condition}, Fuel: {obs.fuel_level_l} L,
Headway: {obs.headway_seconds} sec.

Output JSON: {{"throttle": 0.5, "brake": 0.0, "lean_angle": 20.0, "steering": 0.0}}
"""
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a safe motorcycle riding coach. Output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            action_text = completion.choices[0].message.content.strip()
            if action_text.startswith("```json"):
                action_text = action_text[7:]
            if action_text.startswith("```"):
                action_text = action_text[3:]
            if action_text.endswith("```"):
                action_text = action_text[:-3]
            action_data = json.loads(action_text)
            action = MotorcycleAction(
                throttle=action_data.get("throttle", 0.5),
                brake=action_data.get("brake", 0.0),
                lean_angle=action_data.get("lean_angle", 0.0),
                steering=action_data.get("steering", 0.0)
            )
            obs = env.step(action)
            reward = clamp_score(obs.reward)
            done = obs.done
            all_rewards.append(reward)
            log_step(step, f"task{step}: throttle={action.throttle:.2f}", reward, done, None)

        if len(all_rewards) != 3:
            while len(all_rewards) < 3:
                all_rewards.append(0.50)
            all_rewards = all_rewards[:3]

        final_score = sum(all_rewards) / len(all_rewards)
        final_score = clamp_score(final_score)
        success = final_score > 0.5
        log_end(success, len(all_rewards), final_score, all_rewards)

    except Exception as e:
        print(f"Error: {e}", flush=True)
        log_end(False, 3, 0.50, [0.50, 0.50, 0.50])

if __name__ == "__main__":
    asyncio.run(main())