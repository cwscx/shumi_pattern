import datetime
import torch
import torch.nn as nn
from model import ShumiPatternModel, getBatchData
from device import getDevice


device = getDevice()
print(f"Using device: {device}")

batch_size = 32
iterations = 5000

model = ShumiPatternModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
l1_loss = nn.L1Loss()
cross_entropy_loss = nn.CrossEntropyLoss()


@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    out = {
        "train": {},
        "val": {},
    }
    model.eval()
    outputs = None
    yb = None
    for split in ["train", "val"]:
        losses = {
            "action": torch.zeros(10),
            "milk": torch.zeros(10),
            "milk_amount": torch.zeros(10),
            "daiper": torch.zeros(10),
            "sleep_duration": torch.zeros(10),
            "since_prev_action_duration": torch.zeros(10),
            "time_hour": torch.zeros(10),
            "time_minute": torch.zeros(10),
        }
        for k in range(10):
            xb, yb = getBatchData(split, batch_size=batch_size)
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb)
            action_loss = cross_entropy_loss(
                outputs["action_type"].view(-1, 4),
                yb[:, :, 0:4].argmax(dim=-1).view(-1),
            )
            milk_loss = cross_entropy_loss(
                outputs["milk_type"].view(-1, 4),
                yb[:, :, 4:8].argmax(dim=-1).view(-1),
            )
            milk_amount_loss = l1_loss(
                outputs["milk_amount"].view(-1),
                yb[:, :, 8].view(-1),
            )
            daiper_loss = cross_entropy_loss(
                outputs["daiper_type"].view(-1, 5),
                yb[:, :, 9:14].argmax(dim=-1).view(-1),
            )
            sleep_duration_loss = l1_loss(
                outputs["sleep_duration"].view(-1),
                yb[:, :, 14].view(-1),
            )
            since_prev_action_duration_loss = l1_loss(
                outputs["since_prev_action_duration"].view(-1),
                yb[:, :, 15].view(-1),
            )
            time_hour_loss = l1_loss(
                outputs["time_hour"].view(-1),
                yb[:, :, 16].view(-1),
            )
            time_minute_loss = l1_loss(
                outputs["time_minute"].view(-1),
                yb[:, :, 17].view(-1),
            )
            losses["action"][k] += action_loss.item()
            losses["milk"][k] += milk_loss.item()
            losses["milk_amount"][k] += milk_amount_loss.item()
            losses["daiper"][k] += daiper_loss.item()
            losses["sleep_duration"][k] += sleep_duration_loss.item()
            losses["since_prev_action_duration"][
                k
            ] += since_prev_action_duration_loss.item()
            losses["time_hour"][k] += time_hour_loss.item()
            losses["time_minute"][k] += time_minute_loss.item()

        out[split]["action"] = losses["action"].mean().item()
        out[split]["milk"] = losses["milk"].mean().item()
        out[split]["milk_amount"] = losses["milk_amount"].mean().item()
        out[split]["daiper"] = losses["daiper"].mean().item()
        out[split]["sleep_duration"] = losses["sleep_duration"].mean().item()
        out[split]["since_prev_action_duration"] = (
            losses["since_prev_action_duration"].mean().item()
        )
        out[split]["time_hour"] = losses["time_hour"].mean().item()
        out[split]["time_minute"] = losses["time_minute"].mean().item()
    model.train()
    for k1, v1 in out.items():
        print(f"Step {iter}: {k1} loss:")
        for k2, v2 in v1.items():
            print(f"  - {k2}: {v2:.4f}")
    return out


train_time_start = datetime.datetime.now()
train_time_prev = train_time_start

for iter in range(iterations):
    xb, yb = getBatchData("train", batch_size=batch_size)
    xb = xb.to(device)
    yb = yb.to(device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(xb)
    action_loss = cross_entropy_loss(
        outputs["action_type"].view(-1, 4), yb[:, :, 0:4].argmax(dim=-1).view(-1)
    )
    milk_loss = cross_entropy_loss(
        outputs["milk_type"].view(-1, 4), yb[:, :, 4:8].argmax(dim=-1).view(-1)
    )
    milk_amount_loss = l1_loss(
        outputs["milk_amount"].view(-1),
        yb[:, :, 8].view(-1),
    )
    daiper_loss = cross_entropy_loss(
        outputs["daiper_type"].view(-1, 5), yb[:, :, 9:14].argmax(dim=-1).view(-1)
    )
    sleep_duration_loss = l1_loss(
        outputs["sleep_duration"].view(-1),
        yb[:, :, 14].view(-1),
    )
    since_prev_action_duration_loss = l1_loss(
        outputs["since_prev_action_duration"].view(-1),
        yb[:, :, 15].view(-1),
    )
    time_hour_loss = l1_loss(
        outputs["time_hour"].view(-1),
        yb[:, :, 16].view(-1),
    )
    time_minute_loss = l1_loss(
        outputs["time_minute"].view(-1),
        yb[:, :, 17].view(-1),
    )
    loss = (
        action_loss
        + 0.6 * milk_loss
        + 0.6 * milk_amount_loss
        + 0.6 * daiper_loss
        + 0.6 * sleep_duration_loss
        + 0.6 * since_prev_action_duration_loss
        + 0.4 * time_hour_loss
        + 0.4 * time_minute_loss
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


# Save model.
torch.save(model.state_dict(), "shumi_pattern_model.pth")
