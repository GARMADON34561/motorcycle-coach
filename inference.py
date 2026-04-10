import os
import asyncio
from openai import OpenAI
from models import MotorcycleAction
from server.motorcycle_environment import MotorcycleEnvironment

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY", "dummy")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

def clamp(x): return max(0.01, min(0.99, x))

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = MotorcycleEnvironment()
    print(f"[START] task=motorcycle_coach env=motorcycle_coach model={MODEL_NAME}", flush=True)
    obs = env.reset()
    rewards = []
    for step in range(1, 4):
        prompt = f"Step {step}: Output JSON: {{\"throttle\":0.5, \"brake\":0.0, \"lean_angle\":20.0, \"steering\":0.0}}"
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        txt = completion.choices[0].message.content.strip()
        if txt.startswith("```json"): txt = txt[7:]
        if txt.startswith("```"): txt = txt[3:]
        if txt.endswith("```"): txt = txt[:-3]
        import json
        data = json.loads(txt)
        action = MotorcycleAction(
            throttle=data.get("throttle", 0.5),
            brake=data.get("brake", 0.0),
            lean_angle=data.get("lean_angle", 0.0),
            steering=data.get("steering", 0.0)
        )
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        print(f"[STEP] step={step} action=throttle={action.throttle:.2f} reward={clamp(reward):.2f} done={str(done).lower()} error=null", flush=True)
    final_score = sum(rewards)/3
    final_score = clamp(final_score)
    success = final_score > 0.5
    rewards_str = ",".join(f"{clamp(r):.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps=3 score={final_score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
