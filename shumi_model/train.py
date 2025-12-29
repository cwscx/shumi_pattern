import datetime
import json
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
from shumi_action import (
    Action,
    ShumiAction,
    MilkType,
    DaiperType,
    getAction,
    getDaiperType,
    getDateTime,
    getMilkType,
    getTime,
)


# Gets the device to run the model.
def getDevice() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Loads Shumi actions from the JSON file. Ordered by date and time in an ascending order.
def getShumiActions() -> list[ShumiAction]:
    with open("../shumi_server/shumi_server/shumi.json", "r") as file:
        # Use json.load() to deserialize the file object into a Python dictionary
        shumi_data = json.load(file)
        shumi_patterns = shumi_data["patterns"]
        shumi_actions = []
        for pattern in shumi_patterns:
            day_str = pattern["date"]
            day_parts = day_str.split("/")
            date = datetime.date(
                int(day_parts[0]), int(day_parts[1]), int(day_parts[2])
            )
            shumi_borndate = datetime.date(2025, 9, 6)
            days = (date - shumi_borndate).days

            for pattern_action in pattern["actions"]:
                action = getAction(pattern_action["action"])

                if action == Action.DRINK_MILK:
                    milk_type = getMilkType(pattern_action["type"])
                    volume = int(re.findall(r"\d+", pattern_action["volume"])[0])
                    time = getTime(pattern_action["time_start"])
                    shumi_action = ShumiAction(
                        action,
                        days,
                        time,
                        milk_type=milk_type,
                        milk_amount=volume,
                    )
                    shumi_actions.append(shumi_action)
                # Corner case where Shumi is sleeping and has no time_end.
                elif action == Action.SLEEP and "time_end" in pattern_action:
                    time_start = getTime(pattern_action["time_start"])
                    time_end = getTime(pattern_action["time_end"])
                    datetime_start = getDateTime(days, time_start)
                    datetime_end = getDateTime(days, time_end)
                    duration: datetime.timedelta = datetime_end - datetime_start
                    shumi_action = ShumiAction(
                        action,
                        days,
                        time_start,
                        sleep_duration_min=int(duration.total_seconds() / 60),
                    )
                    shumi_actions.append(shumi_action)
                elif action == Action.CHANGE_DAIPER:
                    daiper_type = getDaiperType(pattern_action["type"])
                    time = getTime(pattern_action["time_start"])
                    shumi_action = ShumiAction(
                        action,
                        days,
                        time,
                        daiper_type=daiper_type,
                    )
                    shumi_actions.append(shumi_action)
        return shumi_actions


def loadData(batch_size: int = 32):
    actions = getShumiActions()
    for action in actions:
        print(action)
    print(len(actions))


loadData()
