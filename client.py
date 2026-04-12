"""Client for Motorcycle Coach environment."""

from openenv.core.env_client import EnvClient
from models import MotorcycleAction, MotorcycleObservation

class MotorcycleCoachClient(EnvClient[MotorcycleAction, MotorcycleObservation]):
    """Client for interacting with the Motorcycle Coach environment."""
    pass