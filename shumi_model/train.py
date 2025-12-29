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


device = getDevice()


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


# Convert a ShumiAction to embedding tensor in shape [8].
# The features are:
#   - Action Type (float)
#   - Milk Type (float)
#   - Milk Amount (float)
#   - Daiper Type (float)
#   - Sleep Duration in minutes (float)
#   - Since Previous Action Duration in minutes (float)
#   - Time Hour (float)
#   - Time Minute (float)
def getActionEmbedding(shumi_action: ShumiAction) -> torch.Tensor:
    action_type_tensor = torch.tensor([shumi_action.action.value], dtype=torch.float32)
    milk_type_tensor = torch.tensor(
        [shumi_action.milk_type.value if shumi_action.milk_type is not None else 0],
        dtype=torch.float32,
    )
    milk_amount_tensor = torch.tensor(
        [shumi_action.milk_amount if shumi_action.milk_amount is not None else 0],
        dtype=torch.float32,
    )
    daiper_type_tensor = torch.tensor(
        [shumi_action.daiper_type.value if shumi_action.daiper_type is not None else 0],
        dtype=torch.float32,
    )
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
    since_prev_action_duration_min_tensor = torch.tensor(
        [shumi_action.since_prev_action_duration.total_seconds() / 60],
        dtype=torch.float32,
    )
    time_hour_tensor = torch.tensor([shumi_action.date_time.hour], dtype=torch.float32)
    time_minute_tensor = torch.tensor(
        [shumi_action.date_time.minute], dtype=torch.float32
    )

    tensor = torch.cat(
        [
            action_type_tensor,
            milk_type_tensor,
            milk_amount_tensor,
            daiper_type_tensor,
            sleep_duration_tensor,
            since_prev_action_duration_min_tensor,
            time_hour_tensor,
            time_minute_tensor,
        ],
        dim=0,
    )

    return tensor


# Gets a batch of action embeddings for training and validation, in shape of [#, block_size, feature_size].
# Where # is the number of data, feature size is the size of each action embedding returned by getActionEmbedding().
def getActionEmbeddings(block_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    actions = getShumiActions()
    actions_tensor = torch.stack([getActionEmbedding(action) for action in actions])

    end_offsets = torch.randint(low=1, high=len(actions) - 1, size=(len(actions) - 1,))
    inputs = torch.stack(
        [
            # If the preceding actions do not have enough length, pad with zeros at the beginning.
            F.pad(
                actions_tensor[max(0, end_offset - block_size - 1) : end_offset],
                (
                    0,
                    0,
                    0,
                    block_size - (end_offset - max(0, end_offset - block_size - 1)),
                ),
            )
            for end_offset in end_offsets
        ]
    )
    outputs = torch.stack([actions_tensor[end_offset] for end_offset in end_offsets])
    return inputs, outputs


# Gets training and validation data.
def getData(
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y = getActionEmbeddings(block_size)
    n = int(len(x) * 0.8)

    return x[:n], y[:n], x[n:], y[n:]


train_x, train_y, test_x, test_y = getData()


# Gets a batch of data for training or validation in one run.
def getBatchData(
    split: str = "train", batch_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    if split == "train":
        x, y = train_x, train_y
    else:
        x, y = test_x, test_y
    indices = torch.randint(len(x), (batch_size,))
    return x[indices], y[indices]


class FeedForwardNN(nn.Module):
    def __init__(self, embedding_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShumiPatternModel(nn.Module):
    def __init__(
        self, input_size: int = 8, embedding_size: int = 128, block_size: int = 16
    ):
        super().__init__()
        # self.input_embedding_table = nn.Embedding(input_size, embedding_size)
        # self.positional_embedding_table = nn.Embedding(block_size, embedding_size)
        self.feed_forward_nn = nn.Sequential(
            nn.Linear(input_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.output = nn.Linear(embedding_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feed_forward_nn(x)
        x = self.output(x)
        return x.mean(dim=1)


model = ShumiPatternModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    out = {}
    model.eval()
    outputs = None
    yb = None
    for split in ["train", "val"]:
        losses = torch.zeros(10)
        for k in range(10):
            xb, yb = getBatchData(split, batch_size=32)
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb)
            loss = loss_fn(outputs, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    print(f"Step {iter}: train loss {out['train']:.4f}, val loss {out['val']:.4f}")
    print(f"{outputs[0]}, {yb[0]}")
    return out


for iter in range(50000):
    xb, yb = getBatchData("train", batch_size=32)
    xb = xb.to(device)
    yb = yb.to(device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(xb)
    loss = loss_fn(outputs, yb)
    loss.backward()
    optimizer.step()

    if iter % 1000 == 0:
        estimate_loss()
