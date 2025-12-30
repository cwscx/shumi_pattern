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
    BIRTHDAY,
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
head_num = 4
input_num = 15
embedding_num = 128
block_size = 16
batch_size = 32
print(f"Using device: {device}")


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
    if enum_val <= 0 or enum_val > num_classes:
        return torch.zeros(num_classes).float()
    return F.one_hot(torch.tensor(enum_val) - 1, num_classes=num_classes).float()


# Convert a ShumiAction to embedding tensor in shape [15].
# The features are:
#   - Action Type (float, 3)
#   - Milk Type (float, 3)
#   - Milk Amount (float, 1)
#   - Daiper Type (float, 4)
#   - Sleep Duration in minutes (float, 1)
#   - Since Previous Action Duration in minutes (float, 1)
#   - Time Hour (float, 1)
#   - Time Minute (float, 1)
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


# Convert an action embedding tensor in shape [8] back to ShumiAction.
def getShumiAction(action_tensor: torch.Tensor) -> ShumiAction:
    action_type_val = action_tensor[0].item()
    action = Action(min(max(round(action_type_val), 1), 3))
    milk_type = None
    milk_amount = None
    daiper_type = None
    days = datetime.datetime.now().date() - BIRTHDAY

    since_prev_action_duration_min = datetime.timedelta(
        minutes=round(action_tensor[5].item())
    )
    time_hour = round(action_tensor[6].item())
    time_minute = round(action_tensor[7].item())

    if action == Action.DRINK_MILK:
        milk_type = MilkType(min(max(round(action_tensor[1].item()), 1), 3))
        milk_amount = max(round(action_tensor[2].item()), 0)
        return ShumiAction(
            action,
            days=days.days,
            time=datetime.time(time_hour, time_minute),
            milk_type=milk_type,
            milk_amount=milk_amount,
            since_prev_action_duration=since_prev_action_duration_min,
        )
    elif action == Action.SLEEP:
        sleep_duration_min = max(round(action_tensor[4].item()), 0)
        return ShumiAction(
            action,
            days=days.days,
            time=datetime.time(time_hour, time_minute),
            sleep_duration_min=sleep_duration_min,
            since_prev_action_duration=since_prev_action_duration_min,
        )
    elif action == Action.CHANGE_DAIPER:
        daiper_type = DaiperType(min(max(round(action_tensor[3].item()), 1), 4))
        return ShumiAction(
            action,
            days=days.days,
            time=datetime.time(time_hour, time_minute),
            daiper_type=daiper_type,
            since_prev_action_duration=since_prev_action_duration_min,
        )
    raise ValueError("Invalid action type")


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
        self.action_type_head = self.head(embedding_size, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.proj(x)
        embedding = self.blocks(embedding)
        embedding = self.ln1(embedding)
        action_outputs = self.action_type_head(embedding)

        return {
            "action_type": action_outputs,
        }

    def head(self, input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, round(input_dim / 4)),
            nn.ReLU(),
            nn.Linear(round(input_dim / 4), output_dim),
        )


model = ShumiPatternModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()
cross_entropy_loss = nn.CrossEntropyLoss()


@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    out = {}
    model.eval()
    outputs = None
    yb = None
    for split in ["train", "val"]:
        losses = torch.zeros(10)
        for k in range(10):
            xb, yb = getBatchData(split, batch_size=batch_size)
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb)
            action_loss = cross_entropy_loss(
                outputs["action_type"].view(-1, 3),
                yb[:, :, 0:3].argmax(dim=-1).view(-1),
            )
            losses[k] += action_loss.item()
        out[split] = losses.mean().item()
    model.train()
    print(f"Step {iter}: train loss {out['train']:.4f}, val loss {out['val']:.4f}")
    return out


train_time_start = datetime.datetime.now()
train_time_prev = train_time_start

for iter in range(5000):
    xb, yb = getBatchData("train", batch_size=batch_size)
    xb = xb.to(device)
    yb = yb.to(device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(xb)
    loss = cross_entropy_loss(
        outputs["action_type"].view(-1, 3), yb[:, :, 0:3].argmax(dim=-1).view(-1)
    )
    loss.backward()
    optimizer.step()

    if iter % 1000 == 0:
        estimate_loss()
        train_time_now = datetime.datetime.now()
        elapsed = train_time_now - train_time_prev
        total_elapsed = train_time_now - train_time_start
        train_time_prev = train_time_now
        print(
            f"Elapsed time for last 1000 iters: {elapsed}, total elapsed time: {total_elapsed}"
        )

with torch.no_grad():
    actions = getShumiActions()[-block_size:]
    last_actions = (
        torch.stack([getActionEmbedding(action) for action in actions])
        .unsqueeze(0)
        .to(device)
    )
    output = model(last_actions)
    action_probs = F.softmax(output["action_type"][:, -1, :], dim=-1)
    action_type_val = torch.argmax(action_probs).item() + 1
    action = Action(action_type_val)
    print(f"Predicted next action: {action}, probabilities: {action_probs}")
