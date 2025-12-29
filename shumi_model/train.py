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
    with open(
        "../shumi_server/shumi_server/shumi.json",
        "r",
    ) as file:
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
                prev_action = shumi_actions[-1] if len(shumi_actions) > 0 else None

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
                        prev_action=prev_action,
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
                        prev_action=prev_action,
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
                        prev_action=prev_action,
                    )
                    shumi_actions.append(shumi_action)
        return shumi_actions


# Convert a ShumiAction to embedding tensor in shape [10].
def getActionEmbedding(shumi_action: ShumiAction) -> torch.Tensor:
    action_type_embedding = nn.Embedding(1, 5)
    milk_type_embedding = nn.Embedding(1, 3)
    milk_amount_tensor = torch.tensor(
        [shumi_action.milk_amount if shumi_action.milk_amount is not None else 0],
        dtype=torch.float32,
    )
    daiper_type_embedding = nn.Embedding(1, 3)
    sleep_duration_tensor = torch.tensor(
        [
            (
                shumi_action.sleep_duration_min
                if shumi_action.sleep_duration_min is not None
                else 0
            )
        ],
        dtype=torch.float32,
    )
    days_tensor = torch.tensor([shumi_action.days], dtype=torch.float32)
    since_prev_action_duration_min_tensor = torch.tensor(
        [shumi_action.since_prev_action_duration.total_seconds() / 60],
        dtype=torch.float32,
    )
    time_hour_tensor = torch.tensor([shumi_action.date_time.hour], dtype=torch.float32)
    time_minute_tensor = torch.tensor(
        [shumi_action.date_time.minute], dtype=torch.float32
    )

    # tensor = torch.cat(
    #     action_type_embedding,
    #     milk_type_embedding,
    #     milk_amount_tensor,
    #     daiper_type_embedding,
    #     sleep_duration_tensor,
    #     days_tensor,
    #     since_prev_action_duration_min_tensor,
    #     time_hour_tensor,
    #     time_minute_tensor,
    # )

    return torch.zeros(10)


# Gets a batch of action embeddings for training and validation, in shape of [#, block_size, feature_size].
# Where # is the number of data, feature size is the size of each action embedding returned by getActionEmbedding().
def getActionEmbeddings(block_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    actions = getShumiActions()
    actions_tensor = torch.stack([getActionEmbedding(action) for action in actions])

    start_offsets = torch.randint(
        high=len(actions) - block_size - 1, size=(len(actions) - block_size - 1,)
    )
    inputs = torch.stack(
        [actions_tensor[start : start + block_size] for start in start_offsets]
    )
    outputs = torch.stack(
        [actions_tensor[start + block_size + 1] for start in start_offsets]
    )
    return inputs, outputs


# Gets training and validation data.
def getData(
    split: str = "train", block_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = getActionEmbeddings(block_size)
    n = int(len(x) * 0.8)

    if split == "train":
        return x[:n], y[:n]
    else:
        return x[n:], y[n:]
    return train_data, val_data


train_x, train_y = getData("train")
test_x, test_y = getData("test")
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
