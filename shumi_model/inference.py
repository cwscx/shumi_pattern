import datetime
import torch
import torch.nn.functional as F
from device import getDevice
from model import ShumiPatternModel, getShumiActions, getActionEmbedding
from shumi_action import Action, MilkType, DaiperType, ShumiAction, BIRTHDAY

block_size = 32
device = getDevice()

model = ShumiPatternModel()
model.load_state_dict(torch.load("shumi_pattern_model.pth"))
model.to(device)


def predict_next_actions(
    num_of_actions: int = 1,
) -> list[tuple[ShumiAction, dict[str, float]]]:
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

            action_probs = F.softmax(output["action_type"][:, -1, :], dim=-1)
            action_type_val = torch.argmax(action_probs).item()
            action = Action(action_type_val)
            action_prob = round(
                action_probs.squeeze()[torch.argmax(action_probs)].item(), 4
            )
            probs["action_type"] = action_prob

            since_prev_action_duration = round(
                output["since_prev_action_duration"][:, -1, :].item()
            )

            if action == Action.UNKNOWN_ACTION or since_prev_action_duration < 0:
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
                sleep_duration = round(output["sleep_duration"][:, -1, :].item())

                if sleep_duration < 0:
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
                milk_type_probs = F.softmax(output["milk_type"][:, -1, :], dim=-1)
                milk_type_val = torch.argmax(milk_type_probs).item()
                milk_type = MilkType(milk_type_val)
                milk_type_prob = round(
                    milk_type_probs.squeeze()[torch.argmax(milk_type_probs)].item(), 4
                )
                probs["milk_type"] = milk_type_prob
                milk_amount = round(output["milk_amount"][:, -1, :].item())

                if milk_type == MilkType.UNKNOWN_ACTION or milk_amount < 0:
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
                daiper_type_probs = F.softmax(output["daiper_type"][:, -1, :], dim=-1)
                daiper_type_val = torch.argmax(daiper_type_probs).item()
                daiper_type = DaiperType(daiper_type_val)

                if daiper_type == DaiperType.UNKNOWN_DAIPER_TYPE:
                    continue

                daiper_type_prob = round(
                    daiper_type_probs.squeeze()[torch.argmax(daiper_type_probs)].item(),
                    4,
                )
                probs["daiper_type"] = daiper_type_prob

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
