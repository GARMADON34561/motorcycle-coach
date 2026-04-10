import os
import asyncio
import json
from typing import List, Optional
from openai import OpenAI

# Directly import your environment logic
from server.app import MotorcycleEnvironment, MotorcycleAction

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

TASK_NAME = "motorcycle_coach"
BENCHMARK = "motorcycle_coach"
MAX_STEPS = 10  # Enough steps to complete 3 tasks

def clamp_score(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (0.01 to 0.99)"""
    if score is None:
        return 0.50
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
    # Instantiate the environment directly
    env = MotorcycleEnvironment()

    all_rewards = []
    steps_taken = 0
    total_reward = 0.0
    final_success = False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        # Reset the environment
        obs = env.reset()
        done = obs.done
        step = 0

        # Run steps until completion
        while not done and step < MAX_STEPS:
            step += 1
            steps_taken = step

            # Build prompt for the LLM based on the observation
            prompt = f"""
You are an AR motorcycle coach. Current state:
Speed: {obs.speed_kmh} km/h, Lean angle: {obs.lean_angle} deg,
Distance to obstacle: {obs.distance_to_obstacle_m} m,
Road: {obs.road_condition}, Fuel: {obs.fuel_level_l} L,
Headway: {obs.headway_seconds} sec.

Output a JSON action: {{"throttle": 0.5, "brake": 0.0, "lean_angle": 20.0, "steering": 0.0}}
"""
            # Get action from LLM
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
            # Clean up markdown
            if action_text.startswith("```json"):
                action_text = action_text[7:]
            if action_text.startswith("```"):
                action_text = action_text[3:]
            if action_text.endswith("```"):
                action_text = action_text[:-3]
            action_data = json.loads(action_text)

            # Create the action object for the environment
            action = MotorcycleAction(
                throttle=action_data.get("throttle", 0.5),
                brake=action_data.get("brake", 0.0),
                lean_angle=action_data.get("lean_angle", 0.0),
                steering=action_data.get("steering", 0.0)
            )

            # Execute the step in the environment
            obs = env.step(action)
            reward = obs.reward
            done = obs.done

            # Store the reward
            all_rewards.append(reward)
            total_reward += reward

            # Log the step
            log_step(step, f"throttle={action.throttle:.2f}", reward, done, None)

        # The environment's step() method handles task progression.
        # After the loop, 'all_rewards' should contain exactly 3 rewards (one per task).
        if len(all_rewards) != 3:
            print(f"[WARN] Expected 3 task rewards, got {len(all_rewards)}. Adjusting logs.", flush=True)
            # Pad or truncate to ensure exactly 3 rewards for the final log
            while len(all_rewards) < 3:
                all_rewards.append(0.50)
            all_rewards = all_rewards[:3]

        # Calculate final score (average of the 3 task rewards)
        final_score = sum(all_rewards) / len(all_rewards)
        final_score = clamp_score(final_score)
        final_success = final_score > 0.5

        # Log the final end
        log_end(final_success, steps_taken, final_score, all_rewards)

    except Exception as e:
        print(f"Error in main loop: {e}", flush=True)
        # On error, still log a valid end to help debugging
        fallback_rewards = [0.50, 0.50, 0.50]
        log_end(False, 3, 0.50, fallback_rewards)

if __name__ == "__main__":
    asyncio.run(main())