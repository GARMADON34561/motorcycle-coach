import os
import asyncio
import json
from openai import OpenAI
from models import MotorcycleAction
from server.motorcycle_environment import MotorcycleEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", "dummy")   # or OPENAI_API_KEY
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

def clamp(x):
    return max(0.01, min(0.99, x))

def safe_action():
    return MotorcycleAction(throttle=0.5, brake=0.0, lean_angle=10.0, steering=0.0)

async def main():
    print(f"[START] task=motorcycle_coach env=motorcycle_coach model={MODEL_NAME}", flush=True)
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        env = MotorcycleEnvironment()
        obs = env.reset()
        rewards = []

        for step in range(1, 4):  # 3 tasks
            action = safe_action()
            try:
                prompt = f"Task {step}: hazard={obs.hazard_type}, distance={obs.hazard_distance:.1f}m, speed={obs.speed:.1f}km/h. Output JSON: {{'throttle':0.5, 'brake':0.0, 'lean_angle':10.0, 'steering':0.0}}"
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=150
                )
                txt = completion.choices[0].message.content.strip()
                # clean markdown
                if txt.startswith("```json"):
                    txt = txt[7:]
                if txt.startswith("```"):
                    txt = txt[3:]
                if txt.endswith("```"):
                    txt = txt[:-3]
                data = json.loads(txt)
                action = MotorcycleAction(
                    throttle=float(data.get("throttle", 0.5)),
                    brake=float(data.get("brake", 0.0)),
                    lean_angle=float(data.get("lean_angle", 10.0)),
                    steering=float(data.get("steering", 0.0))
                )
            except Exception as e:
                print(f"[WARN] Model failed, using fallback: {e}", flush=True)

            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            print(f"[STEP] step={step} action=throttle={action.throttle:.2f} brake={action.brake:.2f} reward={clamp(reward):.2f} done={str(done).lower()} error=null", flush=True)

        final_score = clamp(sum(rewards) / 3)
        success = final_score > 0.5
        rewards_str = ",".join(f"{clamp(r):.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps=3 score={final_score:.3f} rewards={rewards_str}", flush=True)
    except Exception as fatal:
        print(f"[FATAL] {fatal}", flush=True)
        print("[END] success=false steps=0 score=0.010 rewards=0.01,0.01,0.01", flush=True)

if __name__ == "__main__":
    asyncio.run(main())