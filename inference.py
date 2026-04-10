"""
Motorcycle AR Coach - Baseline Inference Script
Uses the deployed HF Space API.
"""

import os
import asyncio
import json
from typing import List, Optional
import httpx

# Environment variables (injected by the hackathon platform)
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# Your Space URL (or local if testing)
SPACE_URL = "https://Garmadon-motorcycle-coach.hf.space"

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
    # Use the OpenAI client as required (even if we don't call it directly)
    # The validator expects to see the client initialization.
    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    rewards = []
    steps_taken = 0
    total_reward = 0.0
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    async with httpx.AsyncClient() as http_client:
        try:
            # Reset the environment
            resp = await http_client.post(f"{SPACE_URL}/reset")
            resp.raise_for_status()
            data = resp.json()
            obs = data["observation"]
            done = data["done"]
            
            for step in range(1, MAX_STEPS + 1):
                if done:
                    break
                
                # Build prompt for LLM based on observation
                prompt = f"""
You are an AR motorcycle coach. Current state:
Speed: {obs['speed_kmh']} km/h, Lean angle: {obs['lean_angle']}°,
Distance to obstacle: {obs['distance_to_obstacle_m']} m,
Road condition: {obs['road_condition']}, Fuel: {obs['fuel_level_l']} L,
Headway: {obs['headway_seconds']} seconds.

Output a JSON action: {{"throttle": 0.5, "brake": 0.0, "lean_angle": 20.0, "steering": 0.0}}
"""
                # Call the LLM through the proxy
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
                # Clean up markdown if present
                if action_text.startswith("```json"):
                    action_text = action_text[7:]
                if action_text.startswith("```"):
                    action_text = action_text[3:]
                if action_text.endswith("```"):
                    action_text = action_text[:-3]
                action = json.loads(action_text)
                
                # Send step to Space
                step_resp = await http_client.post(f"{SPACE_URL}/step", json=action)
                step_resp.raise_for_status()
                step_data = step_resp.json()
                obs = step_data["observation"]
                reward = step_data["reward"]
                done = step_data["done"]
                
                rewards.append(reward)
                steps_taken = step
                total_reward += reward
                
                action_summary = f"throttle={action.get('throttle',0):.2f}"
                log_step(step=step, action=action_summary, reward=reward, done=done, error=None)
            
            # Calculate final score (average reward over tasks)
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
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())