import datetime
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from shumi_action import (
    Action,
    ShumiAction,
    MilkType,
    DaiperType,
    getAction,
    getMilkType,
    getDaiperType,
)


# Gets the device to run the model.
def getDevice() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def loadData(batch_size: int = 32):
    with open("../shumi_server/shumi_server/shumi.json", "r") as file:
        # Use json.load() to deserialize the file object into a Python dictionary
        shumi_data = json.load(file)
        shumi_patterns = shumi_data["patterns"]
        shumi_actinos = []
        for pattern in shumi_patterns[:1]:
            day_str = pattern["date"]
            day_parts = day_str.split("/")
            date = datetime.datetime(
                int(day_parts[0]), int(day_parts[1]), int(day_parts[2])
            )
            shumi_borndate = datetime.datetime(2025, 9, 6)
            days_to_live = (date - shumi_borndate).days

            for pattern_action in pattern["actions"]:
                action = getAction(pattern_action["action"])

                if action == Action.DRINK_MILK:
                    milk_type = getMilkType(pattern_action["type"])
                    volume = int(pattern_action["volume"].replace("ml", ""))
                    print(milk_type)
                    print(volume)


loadData()
