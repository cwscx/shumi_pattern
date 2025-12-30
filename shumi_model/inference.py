import torch
import torch.nn.functional as F
from device import getDevice
from model import ShumiPatternModel, getShumiActions, getActionEmbedding
from shumi_action import Action, MilkType, DaiperType

block_size = 32
device = getDevice()
print(f"Using device: {device}")

model = ShumiPatternModel()
model.load_state_dict(torch.load("shumi_pattern_model.pth"))
model.to(device)


# Predict the next action based on the last block_size actions to evaluate.
with torch.no_grad():
    actions = getShumiActions()[-block_size:]
    last_actions = (
        torch.stack([getActionEmbedding(action) for action in actions])
        .unsqueeze(0)
        .to(device)
    )
    output = model(last_actions)

    action_probs = F.softmax(output["action_type"][:, -1, :], dim=-1)
    action_type_val = torch.argmax(action_probs).item()
    action = Action(action_type_val)
    print(f"Predicted next action: {action}, probabilities: {action_probs}")

    milk_type_probs = F.softmax(output["milk_type"][:, -1, :], dim=-1)
    milk_type_val = torch.argmax(milk_type_probs).item()
    milk_type = MilkType(milk_type_val)
    print(f"Milk type: {milk_type}, probabilities: {milk_type_probs}")
    milk_amount = round(output["milk_amount"][:, -1, :].item())
    print(f"Predicted milk amount (ml): {milk_amount}")

    daiper_type_probs = F.softmax(output["daiper_type"][:, -1, :], dim=-1)
    dayper_type_val = torch.argmax(daiper_type_probs).item()
    daiper_type = DaiperType(dayper_type_val)
    print(f"Daiper type: {daiper_type}, probabilities: {daiper_type_probs}")

    sleep_duration = round(output["sleep_duration"][:, -1, :].item())
    print(f"Predicted sleep duration (minutes): {sleep_duration}")

    since_prev_action_duration = round(
        output["since_prev_action_duration"][:, -1, :].item()
    )
    print(
        f"Predicted since previous action duration (minutes): {since_prev_action_duration}"
    )
    time_hour = round(output["time_hour"][:, -1, :].item())
    time_minute = round(output["time_minute"][:, -1, :].item())
    print(f"Predicted time: {time_hour:02d}:{time_minute:02d}")
