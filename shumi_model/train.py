import datetime
import json
import os
import math
import torch
import torch.nn as nn
from model import ShumiPatternModel, getBatchData, batch_size
from device import getDevice
from shumi_action import Action


device = getDevice()
print(f"Using device: {device}")

iterations = 10000
eval_every_step = 100
max_iter_wait = 15  # max iterations to wait if no improvement.

model = ShumiPatternModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
l1_loss = nn.L1Loss()
cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.02)


@torch.no_grad()
def estimate_loss() -> tuple[list[str], dict[str, dict[str, float]]]:
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
            "time_sin": torch.zeros(10),
            "time_cos": torch.zeros(10),
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
            action_target = yb[:, -1, 0:4].argmax(dim=-1)
            action_loss = cross_entropy_loss(
                outputs["action_type"][:, -1, 0:4].view(-1, 4),
                action_target.view(-1),
            )

            milk_mask = (action_target == Action.DRINK_MILK.value).float()
            milk_loss = cross_entropy_loss(
                outputs["milk_type"][:, -1, :].view(-1, 4),
                yb[:, -1, 4:8].argmax(dim=-1).view(-1),
            )
            milk_loss = (milk_loss * milk_mask).sum() / (milk_mask.sum().clamp_min(1.0))
            milk_amount_loss = l1_loss(
                outputs["milk_amount"][:, -1, :].view(-1),
                yb[:, -1, 8].view(-1),
            )
            milk_amount_loss = (milk_amount_loss * milk_mask).sum() / (
                milk_mask.sum().clamp_min(1.0)
            )

            daiper_mask = (action_target == Action.CHANGE_DAIPER.value).float()
            daiper_loss = cross_entropy_loss(
                outputs["daiper_type"][:, -1, :].view(-1, 5),
                yb[:, -1, 9:14].argmax(dim=-1).view(-1),
            )
            daiper_loss = (daiper_loss * daiper_mask).sum() / (
                daiper_mask.sum().clamp_min(1.0)
            )

            sleep_mask = (action_target == Action.SLEEP.value).float()
            sleep_duration_loss = l1_loss(
                outputs["sleep_duration"][:, -1, :].view(-1),
                yb[:, -1, 14].view(-1),
            )
            sleep_duration_loss = (sleep_duration_loss * sleep_mask).sum() / (
                sleep_mask.sum().clamp_min(1.0)
            )

            since_prev_action_duration_loss = l1_loss(
                outputs["since_prev_action_duration"][:, -1, :].view(-1),
                yb[:, -1, 15].view(-1),
            )
            time_sin_loss = l1_loss(
                outputs["time_sin"][:, -1, :].view(-1), yb[:, -1, 16].view(-1)
            )
            time_cos_loss = l1_loss(
                outputs["time_cos"][:, -1, :].view(-1), yb[:, -1, 17].view(-1)
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
            losses["milk"][k] += milk_loss.item()
            losses["milk_amount"][k] += milk_amount_loss.item()
            losses["daiper"][k] += daiper_loss.item()
            losses["sleep_duration"][k] += sleep_duration_loss.item()
            losses["since_prev_action_duration"][
                k
            ] += since_prev_action_duration_loss.item()
            losses["time_sin"][k] += time_sin_loss.item()
            losses["time_cos"][k] += time_cos_loss.item()
            losses["weeks"][k] += weeks_loss.item()

            accuracies["action"][k] += action_acc

        out[split]["action"] = losses["action"].mean().item()
        out[split]["action_acc"] = accuracies["action"].mean().item()
        out[split]["drink_milk_recall"] = accuracies["drink_milk_recall"].mean().item()
        out[split]["sleep_recall"] = accuracies["sleep_recall"].mean().item()
        out[split]["change_daiper_recall"] = (
            accuracies["change_daiper_recall"].mean().item()
        )

        out[split]["milk"] = losses["milk"].mean().item()
        out[split]["milk_amount"] = losses["milk_amount"].mean().item()
        out[split]["daiper"] = losses["daiper"].mean().item()
        out[split]["sleep_duration"] = losses["sleep_duration"].mean().item()
        out[split]["since_prev_action_duration"] = (
            losses["since_prev_action_duration"].mean().item()
        )
        out[split]["time_sin"] = losses["time_sin"].mean().item()
        out[split]["time_cos"] = losses["time_cos"].mean().item()
        out[split]["weeks"] = losses["weeks"].mean().item()

    model.train()

    losses = []
    for k1, v1 in out.items():
        losses.append(f"Step {iter}: {k1} loss:")
        print(losses[-1])
        for k2, v2 in v1.items():
            losses.append(f"  - {k2}: {v2:.4f}")
            print(losses[-1])
    return losses, out


train_time_start = datetime.datetime.now()
train_time_prev = train_time_start

best_loss = 1000
best_loss_iter = 0
model_metadata = {}

with open(
    os.path.dirname(__file__) + "/model.json",
    "r",
) as file:
    # Use json.load() to deserialize the file object into a Python dictionary
    model_metadata = json.load(file)
    best_loss = model_metadata["best_loss_score"]

for iter in range(iterations + 1):
    xb, yb = getBatchData("train", batch_size=batch_size)
    xb = xb.to(device)
    yb = yb.to(device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(xb)
    action_target = yb[:, -1, 0:4].argmax(dim=-1)
    action_loss = cross_entropy_loss(
        outputs["action_type"][:, -1, :].view(-1, 4), action_target.view(-1)
    )

    # Only calculate milk loss if the type is drink milk.
    milk_mask = (action_target == Action.DRINK_MILK.value).float()
    milk_loss = cross_entropy_loss(
        outputs["milk_type"][:, -1, :].view(-1, 4),
        yb[:, -1, 4:8].argmax(dim=-1).view(-1),
    )
    milk_loss = (milk_loss * milk_mask).sum() / (milk_mask.sum().clamp_min(1.0))
    milk_amount_loss = l1_loss(
        outputs["milk_amount"][:, -1, :].view(-1),
        yb[:, -1, 8].view(-1),
    )
    milk_amount_loss = (milk_amount_loss * milk_mask).sum() / (
        milk_mask.sum().clamp_min(1.0)
    )

    # Only calculate diaper loss if the type is change daiper.
    daiper_mask = (action_target == Action.CHANGE_DAIPER.value).float()
    daiper_loss = cross_entropy_loss(
        outputs["daiper_type"][:, -1, :].view(-1, 5),
        yb[:, -1, 9:14].argmax(dim=-1).view(-1),
    )
    daiper_loss = (daiper_loss * daiper_mask).sum() / (daiper_mask.sum().clamp_min(1.0))

    # Only calculate sleep loss if the type is sleep.
    sleep_mask = (action_target == Action.SLEEP.value).float()
    sleep_duration_loss = l1_loss(
        outputs["sleep_duration"][:, -1, :].view(-1),
        yb[:, -1, 14].view(-1),
    )
    sleep_duration_loss = (sleep_duration_loss * sleep_mask).sum() / (
        sleep_mask.sum().clamp_min(1.0)
    )

    # Time feature losses are always calculated.
    since_prev_action_duration_loss = l1_loss(
        outputs["since_prev_action_duration"][:, -1, :].view(-1),
        yb[:, -1, 15].view(-1),
    )
    time_sin_loss = l1_loss(
        outputs["time_sin"][:, -1, :].view(-1),
        yb[:, -1, 16].view(-1),
    )
    time_cos_loss = l1_loss(
        outputs["time_cos"][:, -1, :].view(-1),
        yb[:, -1, 17].view(-1),
    )
    weeks_loss = l1_loss(outputs["weeks"][:, -1, :].view(-1), yb[:, -1, 18].view(-1))

    loss = (
        1.0 * action_loss
        + 0.2 * milk_loss
        + 0.2 * milk_amount_loss
        + 0.2 * daiper_loss
        + 0.2 * sleep_duration_loss
        + 0.4 * since_prev_action_duration_loss
        + 0.4 * time_sin_loss
        + 0.4 * time_cos_loss
        + 0.4 * weeks_loss
    )
    loss.backward()
    optimizer.step()

    if iter % eval_every_step == 0:
        losses, out = estimate_loss()
        train_time_now = datetime.datetime.now()
        elapsed = train_time_now - train_time_prev
        total_elapsed = train_time_now - train_time_start
        train_time_prev = train_time_now

        print(
            f"Elapsed time for last 1000 iters: {elapsed}, total elapsed time: {total_elapsed}"
        )

        action_accuracy = out["val"]["action_acc"]
        action_accuracy_penalty = -math.log(max(action_accuracy, 1e-6))
        min_action_recall = min(
            out["val"]["drink_milk_recall"],
            out["val"]["sleep_recall"],
            out["val"]["change_daiper_recall"],
        )
        recall_pen = max(0.7 - min_action_recall, 0.0)

        loss = (
            1.0 * out["val"]["action"]
            + 2.0 * action_accuracy_penalty
            + 3.0 * recall_pen
            + 0.1 * out["val"]["milk"]
            + 0.1 * out["val"]["milk_amount"]
            + 0.1 * out["val"]["daiper"]
            + 0.1 * out["val"]["sleep_duration"]
            + 0.2 * out["val"]["since_prev_action_duration"]
            + 0.2 * out["val"]["time_sin"]
            + 0.2 * out["val"]["time_cos"]
            + 0.2 * out["val"]["weeks"]
        )

        improved = loss < (best_loss + 1e-4)  # 1e-4 防抖
        if improved:
            print(
                f"""
                Save model at step {iter} with improved accuracy from {best_loss:.4f} to {loss:.4f}, where
                action loss: {out["val"]["action"]}
                action accuracy: {action_accuracy, action_accuracy_penalty}
                min action recall: {min_action_recall}
                milk type loss: {out["val"]["milk"]}
                milk amount loss: {out["val"]["milk_amount"]}
                daiper loss: {out["val"]["daiper"]}
                sleep duration loss: {out["val"]["sleep_duration"]}
                since prev action loss: {out["val"]["since_prev_action_duration"]}
                time sin loss: {out["val"]["time_sin"]}
                time cos loss: {out["val"]["time_cos"]}
                weeks loss: {out["val"]["weeks"]}
                """
            )
            best_loss = loss
            best_loss_iter = iter

            torch.save(
                model.state_dict(),
                os.path.dirname(__file__) + "/shumi_pattern_model.pth",
            )

            with open(
                os.path.dirname(__file__) + "/model.json",
                "w",
            ) as file:
                model_metadata["best_loss_score"] = best_loss
                json.dump(model_metadata, file, indent=4)

        # Early break if no obvious improvement.
        elif iter - best_loss_iter >= max_iter_wait * eval_every_step:
            break
        else:
            print(f"loss is {loss}.")
