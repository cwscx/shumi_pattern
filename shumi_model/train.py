import datetime
import os
import torch
import torch.nn as nn
from model import ShumiPatternModel, getBatchData, batch_size
from device import getDevice


device = getDevice()
print(f"Using device: {device}")

iterations = 1000

model = ShumiPatternModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
l1_loss = nn.L1Loss()
cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.02)


@torch.no_grad()
def estimate_loss() -> list[str]:
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
            # "milk": torch.zeros(10),
            # "milk_amount": torch.zeros(10),
            # "daiper": torch.zeros(10),
            # "sleep_duration": torch.zeros(10),
            "since_prev_action_duration": torch.zeros(10),
            "time_hour": torch.zeros(10),
            "time_minute": torch.zeros(10),
            "weeks": torch.zeros(10),
        }
        accuracies = {
            "action": torch.zeros(10),
            "drink_milk_recall": torch.zeros(10),
            "sleep_recall": torch.zeros(10),
            "change_daiper_recall": torch.zeros(10),
        }
        for k in range(10):
            xb, yb = getBatchData(split, batch_size=batch_size)
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb)
            action_loss = cross_entropy_loss(
                outputs["action_type"][:, -1, 0:4].view(-1, 4),
                yb[:, -1, 0:4].argmax(dim=-1).view(-1),
            )

            # milk_loss = cross_entropy_loss(
            #     outputs["milk_type"].view(-1, 4),
            #     yb[:, :, 4:8].argmax(dim=-1).view(-1),
            # )
            # milk_amount_loss = l1_loss(
            #     outputs["milk_amount"].view(-1),
            #     yb[:, :, 8].view(-1),
            # )
            # daiper_loss = cross_entropy_loss(
            #     outputs["daiper_type"].view(-1, 5),
            #     yb[:, :, 9:14].argmax(dim=-1).view(-1),
            # )
            # sleep_duration_loss = l1_loss(
            #     outputs["sleep_duration"].view(-1),
            #     yb[:, :, 14].view(-1),
            # )
            since_prev_action_duration_loss = l1_loss(
                outputs["since_prev_action_duration"][:, -1, :].view(-1),
                yb[:, -1, 15].view(-1),
            )
            time_hour_loss = l1_loss(
                outputs["time_hour"][:, -1, :].view(-1), yb[:, -1, 16].view(-1)
            )
            time_minute_loss = l1_loss(
                outputs["time_minute"][:, -1, :].view(-1), yb[:, -1, 17].view(-1)
            )
            weeks_loss = l1_loss(
                outputs["weeks"][:, -1, :].view(-1), yb[:, -1, 18].view(-1)
            )

            action_pred = outputs["action_type"][:, -1, :].argmax(dim=-1).view(-1)
            action_target = yb[:, -1, 0:4].argmax(dim=-1).view(-1)
            action_acc = (action_pred == action_target).float().mean().item()
            for c in range(1, 4):
                tp = (
                    ((action_pred.view(-1) == c) & (action_target.view(-1) == c))
                    .sum()
                    .item()
                )
                fn = (
                    ((action_pred.view(-1) != c) & (action_target.view(-1) == c))
                    .sum()
                    .item()
                )
                recall = tp / max(tp + fn, 1)
                if c == 1:
                    accuracies["drink_milk_recall"][k] += recall
                elif c == 2:
                    accuracies["sleep_recall"][k] += recall
                elif c == 3:
                    accuracies["change_daiper_recall"][k] += recall

            losses["action"][k] += action_loss.item()
            # losses["milk"][k] += milk_loss.item()
            # losses["milk_amount"][k] += milk_amount_loss.item()
            # losses["daiper"][k] += daiper_loss.item()
            # losses["sleep_duration"][k] += sleep_duration_loss.item()
            losses["since_prev_action_duration"][
                k
            ] += since_prev_action_duration_loss.item()
            losses["time_hour"][k] += time_hour_loss.item()
            losses["time_minute"][k] += time_minute_loss.item()
            losses["weeks"][k] += weeks_loss.item()

            accuracies["action"][k] += action_acc

        out[split]["action"] = losses["action"].mean().item()
        out[split]["action_acc"] = accuracies["action"].mean().item()
        out[split]["drink_milk_recall"] = accuracies["drink_milk_recall"].mean().item()
        out[split]["sleep_recall"] = accuracies["sleep_recall"].mean().item()
        out[split]["change_daiper_recall"] = (
            accuracies["change_daiper_recall"].mean().item()
        )

        # out[split]["milk"] = losses["milk"].mean().item()
        # out[split]["milk_amount"] = losses["milk_amount"].mean().item()
        # out[split]["daiper"] = losses["daiper"].mean().item()
        # out[split]["sleep_duration"] = losses["sleep_duration"].mean().item()
        out[split]["since_prev_action_duration"] = (
            losses["since_prev_action_duration"].mean().item()
        )
        out[split]["time_hour"] = losses["time_hour"].mean().item()
        out[split]["time_minute"] = losses["time_minute"].mean().item()
        out[split]["weeks"] = losses["weeks"].mean().item()

    model.train()

    losses = []
    for k1, v1 in out.items():
        losses.append(f"Step {iter}: {k1} loss:")
        print(losses[-1])
        for k2, v2 in v1.items():
            losses.append(f"  - {k2}: {v2:.4f}")
            print(losses[-1])
    return losses


train_time_start = datetime.datetime.now()
train_time_prev = train_time_start

for iter in range(iterations + 1):
    xb, yb = getBatchData("train", batch_size=batch_size)
    xb = xb.to(device)
    yb = yb.to(device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(xb)
    action_loss = cross_entropy_loss(
        outputs["action_type"][:, -1, :].view(-1, 4),
        yb[:, -1, 0:4].argmax(dim=-1).view(-1),
    )
    # milk_loss = cross_entropy_loss(
    #     outputs["milk_type"].view(-1, 4), yb[:, :, 4:8].argmax(dim=-1).view(-1)
    # )
    # milk_amount_loss = l1_loss(
    #     outputs["milk_amount"].view(-1),
    #     yb[:, :, 8].view(-1),
    # )
    # daiper_loss = cross_entropy_loss(
    #     outputs["daiper_type"].view(-1, 5), yb[:, :, 9:14].argmax(dim=-1).view(-1)
    # )
    # sleep_duration_loss = l1_loss(
    #     outputs["sleep_duration"].view(-1),
    #     yb[:, :, 14].view(-1),
    # )
    since_prev_action_duration_loss = l1_loss(
        outputs["since_prev_action_duration"][:, -1, :].view(-1),
        yb[:, -1, 15].view(-1),
    )
    time_hour_loss = l1_loss(
        outputs["time_hour"][:, -1, :].view(-1),
        yb[:, -1, 16].view(-1),
    )
    time_minute_loss = l1_loss(
        outputs["time_minute"][:, -1, :].view(-1),
        yb[:, -1, 17].view(-1),
    )
    weeks_loss = l1_loss(outputs["weeks"][:, -1, :].view(-1), yb[:, -1, 18].view(-1))

    loss = (
        1.0 * action_loss
        # + 0.6 * milk_loss
        # + 0.6 * milk_amount_loss
        # + 0.6 * daiper_loss
        # + 0.6 * sleep_duration_loss
        + 0.6 * since_prev_action_duration_loss
        + 0.6 * time_hour_loss
        + 0.6 * time_minute_loss
        + 0.6 * weeks_loss
    )
    loss.backward()
    optimizer.step()

    if iter % 200 == 0:
        estimate_loss()
        train_time_now = datetime.datetime.now()
        elapsed = train_time_now - train_time_prev
        total_elapsed = train_time_now - train_time_start
        train_time_prev = train_time_now

        print(
            f"Elapsed time for last 1000 iters: {elapsed}, total elapsed time: {total_elapsed}"
        )

# Save model.
torch.save(model.state_dict(), os.path.dirname(__file__) + "/shumi_pattern_model.pth")
