import os
from fastapi import FastAPI
from openenv.core.env_server.fastapi_env_server import FastAPIEnvServer
from server.motorcycle_environment import MotorcycleEnvironment

app = FastAPI()

# Create and mount OpenEnv server
env_server = FastAPIEnvServer(MotorcycleEnvironment)
app.mount("/v1", env_server.app)

# Add a simple root health endpoint
@app.get("/")
async def health():
    return {"status": "running", "environment": "motorcycle-coach"}