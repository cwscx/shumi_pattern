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

head_num = 4
input_num = 18
embedding_num = 128
block_size = 32
batch_size = 32


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


def one_hot(enum_val: int, num_classes: int):
    if enum_val < 0 or enum_val >= num_classes:
        return torch.zeros(num_classes).float()
    return F.one_hot(torch.tensor(enum_val), num_classes=num_classes).float()


# Convert a ShumiAction to embedding tensor in shape [18].
# The features are:
#   - Action Type (float, 4) [0:4]
#   - Milk Type (float, 4) [4:8]
#   - Milk Amount (float, 1) [8]
#   - Daiper Type (float, 5) [9:14]
#   - Sleep Duration in minutes (float, 1) [14]
#   - Since Previous Action Duration in minutes (float, 1) [15]
#   - Time Hour (float, 1) [16]
#   - Time Minute (float, 1) [17]
def getActionEmbedding(shumi_action: ShumiAction) -> torch.Tensor:
    action_type_tensor = one_hot(shumi_action.action.value, len(Action))
    milk_type_tensor = one_hot(
        shumi_action.milk_type.value if shumi_action.milk_type is not None else 0,
        len(MilkType),
    )
    daiper_type_tensor = one_hot(
        shumi_action.daiper_type.value if shumi_action.daiper_type is not None else 0,
        len(DaiperType),
    )
    milk_amount_tensor = torch.tensor(
        [shumi_action.milk_amount if shumi_action.milk_amount is not None else 0],
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
def getActionEmbeddings(
    block_size: int = block_size,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    outputs = torch.stack(
        [
            # If the preceding actions do not have enough length, pad with zeros at the beginning.
            F.pad(
                actions_tensor[max(1, end_offset - block_size) : end_offset + 1],
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
    return inputs, outputs


# Gets training and validation data.
def getData(
    block_size: int = block_size,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y = getActionEmbeddings(block_size)
    n = int(len(x) * 0.8)

    return x[:n], y[:n], x[n:], y[n:]


train_x, train_y, test_x, test_y = getData()


# Gets a batch of data for training or validation in one run.
def getBatchData(
    split: str = "train", batch_size: int = batch_size
) -> tuple[torch.Tensor, torch.Tensor]:
    if split == "train":
        x, y = train_x, train_y
    else:
        x, y = test_x, test_y
    indices = torch.randint(len(x), (batch_size,))
    return x[indices], y[indices]


class Head(nn.Module):
    def __init__(self, head_size: int, embedding_num: int = embedding_num):
        super().__init__()
        self.key = nn.Linear(embedding_num, head_size, bias=False)
        self.query = nn.Linear(embedding_num, head_size, bias=False)
        self.value = nn.Linear(embedding_num, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * (embedding_num**-0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, num_heads * head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, embedding_num)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForwardNN(nn.Module):
    def __init__(self, embedding_size: int = embedding_num):
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


class Block(nn.Module):
    def __init__(
        self,
        head_num: int = head_num,
        embedding_size: int = embedding_num,
    ):
        super().__init__()
        self.multi_head = MultiHeadAttention(head_num, embedding_size // head_num)
        self.ffwd = FeedForwardNN(embedding_size)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.multi_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class ShumiPatternModel(nn.Module):
    def __init__(
        self,
        input_size: int = input_num,
        embedding_size: int = embedding_num,
    ):
        super().__init__()
        self.proj = nn.Linear(input_size, embedding_size)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.blocks = nn.Sequential(
            Block(head_num, embedding_size),
            Block(head_num, embedding_size),
            Block(head_num, embedding_size),
        )
        self.action_type_head = self.head(embedding_size, 4)
        self.milk_type_head = self.head(embedding_size, 4)
        self.milk_amount_head = self.head(embedding_size, 1)
        self.daiper_type_head = self.head(embedding_size, 5)
        self.sleep_duration_head = self.head(embedding_size, 1)
        self.since_prev_action_duration_head = self.head(embedding_size, 1)
        self.time_hour_head = self.head(embedding_size, 1)
        self.time_minute_head = self.head(embedding_size, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.proj(x)
        embedding = self.blocks(embedding)
        embedding = self.ln1(embedding)

        # Output heads.
        action_outputs = self.action_type_head(embedding)
        milk_outputs = self.milk_type_head(embedding)
        milk_amount_outputs = self.milk_amount_head(embedding)
        daiper_outputs = self.daiper_type_head(embedding)
        sleep_duration_outputs = self.sleep_duration_head(embedding)
        since_prev_action_duration_outputs = self.since_prev_action_duration_head(
            embedding
        )
        time_hour_outputs = self.time_hour_head(embedding)
        time_minute_outputs = self.time_minute_head(embedding)

        return {
            "action_type": action_outputs,
            "milk_type": milk_outputs,
            "milk_amount": milk_amount_outputs,
            "daiper_type": daiper_outputs,
            "sleep_duration": sleep_duration_outputs,
            "since_prev_action_duration": since_prev_action_duration_outputs,
            "time_hour": time_hour_outputs,
            "time_minute": time_minute_outputs,
        }

    def head(self, input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, round(input_dim / 4)),
            nn.ReLU(),
            nn.Linear(round(input_dim / 4), output_dim),
        )
