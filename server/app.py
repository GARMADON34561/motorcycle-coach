import uvicorn
from openenv.core.env_server import create_app
from models import MotorcycleAction, MotorcycleObservation
from .motorcycle_environment import MotorcycleEnvironment

app = create_app(MotorcycleEnvironment, MotorcycleAction, MotorcycleObservation, env_name="motorcycle_coach")

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()