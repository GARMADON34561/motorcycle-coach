from openenv.core.client import EnvClient   # correct import
from models import MotorcycleAction, MotorcycleObservation

class MotorcycleCoachClient(EnvClient[MotorcycleAction, MotorcycleObservation]):
    pass