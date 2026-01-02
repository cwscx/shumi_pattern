import datetime
import os
import torch
import torch.nn.functional as F
from device import getDevice
from model import (
    ShumiPatternModel,
    getShumiActions,
    getActionEmbedding,
    block_size,
    since_prev_action_duration_std,
    milk_amount_std,
    sleep_duration_std,
)
from shumi_action import Action, MilkType, DaiperType, ShumiAction, BIRTHDAY

device = getDevice()

repetition_penalty = 2.0
top_k = 2
temperature = 0.8
model = ShumiPatternModel()
model.load_state_dict(
    torch.load(os.path.dirname(__file__) + "/shumi_pattern_model.pth")
)
model.to(device)
model.eval()


def predict_next_actions(
    num_of_actions: int = 1,
) -> list[tuple[ShumiAction, dict[str, torch.Tensor]]]:
    actions = getShumiActions()[-block_size:]
    predictions = []
    for _ in range(num_of_actions):
        probs = {}
        with torch.no_grad():
            last_actions = (
                torch.stack([getActionEmbedding(action) for action in actions])
                .unsqueeze(0)
                .to(device)
            )

            output = model(last_actions)

            action_logits = output["action_type"][:, -1, :]
            # Last block_size step penality.
            for i in range(1, block_size + 1):
                last_action_type = actions[-i].action.value
                action_logits[:, last_action_type] -= repetition_penalty * (
                    1 / (i**0.8)
                )
            action_probs = F.softmax(action_logits / temperature, dim=-1)
            action_probs[:, 0] = 0  # Mask the unknown action type to 0 probability.

            action_topk_probs, action_topk_idx = torch.topk(action_probs, k=top_k)

            action_type_sample = torch.multinomial(action_topk_probs, num_samples=1)
            action_type_val = action_topk_idx.squeeze()[
                action_type_sample.squeeze()
            ].item()
            action = Action(action_type_val)
            probs["action_type"] = action_probs

            since_prev_action_duration = round(
                output["since_prev_action_duration"][:, -1, :].item()
                * since_prev_action_duration_std
            )

            if action == Action.UNKNOWN_ACTION or since_prev_action_duration < 0:
                print(f"unexpcted action {action} or {since_prev_action_duration}")
                continue

            last_event = actions[-1]
            last_event_datetime = datetime.datetime(
                year=last_event.date_time.year,
                month=last_event.date_time.month,
                day=last_event.date_time.day,
                hour=last_event.date_time.hour,
                minute=last_event.date_time.minute,
            )
            new_event_datetime = last_event_datetime + datetime.timedelta(
                minutes=round(since_prev_action_duration)
            )
            if (
                last_event.sleep_duration_min is not None
                and last_event.action == Action.SLEEP
            ):
                new_event_datetime += datetime.timedelta(
                    minutes=last_event.sleep_duration_min
                )
            days = (new_event_datetime.date() - BIRTHDAY).days

            if action == Action.SLEEP:
                sleep_duration = round(
                    output["sleep_duration"][:, -1, :].item() * sleep_duration_std
                )

                if sleep_duration <= 0:
                    print(f"Unexpected sleep duration {sleep_duration}.")
                    continue

                predicted_action = ShumiAction(
                    action=action,
                    days=days,
                    time=new_event_datetime.time(),
                    prev_action=actions[-1],
                    since_prev_action_duration=datetime.timedelta(
                        minutes=since_prev_action_duration
                    ),
                    sleep_duration_min=sleep_duration,
                )
                predictions.append((predicted_action, probs))
            elif action == Action.DRINK_MILK:
                milk_logits = output["milk_type"][:, -1, :]
                milk_type_probs = F.softmax(milk_logits / temperature, dim=-1)
                milk_type_probs[:, 0] = (
                    0  # Mask the unknown milk type to 0 probability.
                )

                milk_topk_probs, milk_topk_idx = torch.topk(milk_type_probs, k=top_k)
                milk_type_sample = torch.multinomial(milk_topk_probs, num_samples=1)
                milk_type_val = milk_topk_idx.squeeze()[
                    milk_type_sample.squeeze()
                ].item()
                milk_type = MilkType(milk_type_val)
                probs["milk_type"] = milk_type_probs

                milk_amount = round(
                    output["milk_amount"][:, -1, :].item() * milk_amount_std
                )

                if milk_type == MilkType.UNKNOWN_MILK_TYPE or milk_amount <= 0:
                    print(
                        f"Unexpected milk type {milk_type} or milk amount {milk_amount}."
                    )
                    continue

                predicted_action = ShumiAction(
                    action=action,
                    days=days,
                    time=new_event_datetime.time(),
                    prev_action=actions[-1],
                    since_prev_action_duration=datetime.timedelta(
                        minutes=since_prev_action_duration
                    ),
                    milk_type=milk_type,
                    milk_amount=milk_amount,
                )
                predictions.append((predicted_action, probs))
            elif action == Action.CHANGE_DAIPER:
                daiper_logits = output["daiper_type"][:, -1, :]
                daiper_type_probs = F.softmax(daiper_logits / temperature, dim=-1)
                daiper_type_probs[:, 0] = (
                    0  # Mask the unknown daiper type to 0 probability.
                )
                daiper_topk_probs, daiper_topk_idx = torch.topk(
                    daiper_type_probs, k=top_k
                )
                daiper_type_sample = torch.multinomial(daiper_topk_probs, num_samples=1)
                daiper_type_val = daiper_topk_idx.squeeze()[
                    daiper_type_sample.squeeze()
                ].item()
                daiper_type = DaiperType(daiper_type_val)
                probs["daiper_type"] = daiper_type_probs

                if daiper_type == DaiperType.UNKNOWN_DAIPER_TYPE:
                    print(f"Unexpected daiper type {daiper_type}.")
                    continue

                predicted_action = ShumiAction(
                    action=action,
                    days=days,
                    time=new_event_datetime.time(),
                    prev_action=actions[-1],
                    since_prev_action_duration=datetime.timedelta(
                        minutes=since_prev_action_duration
                    ),
                    daiper_type=daiper_type,
                )
                predictions.append((predicted_action, probs))
            actions.append(predictions[-1][0])
            actions = actions[1:]
    return predictions
