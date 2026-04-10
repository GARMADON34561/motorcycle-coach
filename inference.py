"""
Motorcycle AR Coach - Baseline Inference Script
"""

import os
import asyncio
import json
from typing import List, Optional
from openai import OpenAI

# Import the environment from app.py
from app import MotorcycleEnvironment, MotorcycleAction

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

TASK_NAME = "motorcycle_coach"
BENCHMARK = "motorcycle_coach"
MAX_STEPS = 10

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = MotorcycleEnvironment()
    
    rewards = []
    steps_taken = 0
    total_reward = 0.0
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        obs = env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break
            
            prompt = f"""
You are an AR motorcycle coach. Current state:
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
            # Clean markdown
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
            reward = obs.reward
            done = obs.done
            
            rewards.append(reward)
            steps_taken = step
            total_reward += reward
            
            log_step(step, f"throttle={action.throttle:.2f}", reward, done, None)
            if done:
                break
        
        score = total_reward / len(rewards) if rewards else 0.0
        score = max(0.01, min(0.99, score))
        success = score >= 0.5
        
    except Exception as e:
        print(f"Error: {e}", flush=True)
        success = False
        score = 0.0
        steps_taken = 0
        rewards = []
    finally:
        log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    asyncio.run(main())