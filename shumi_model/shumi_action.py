import datetime
from enum import Enum
from typing import Optional

BIRTHDAY = datetime.date(2025, 9, 6)


class Action(Enum):
    UNKNOWN_ACTION = 0
    DRINK_MILK = 1
    SLEEP = 2
    CHANGE_DAIPER = 3


class MilkType(Enum):
    UNKNOWN_MILK_TYPE = 0
    BOTTLE_BREAST_MILK = 1
    BOTTLE_FORMULA_MILK = 2
    BREAST_FEED = 3


class DaiperType(Enum):
    UNKNOWN_DAIPER_TYPE = 0
    PEE = 1
    POO = 2
    PEE_AND_POO = 3
    CLEAN = 4


class ShumiAction:

    # Member variables.
    days: int
    date_time: datetime.datetime
    since_prev_action_duration: datetime.timedelta
    action: Action
    milk_amount: Optional[int] = None
    milk_type: Optional[MilkType] = None
    daiper_type: Optional[DaiperType] = None
    sleep_duration_min: Optional[int] = None

    def __init__(
        self,
        action: Action,
        days: int,
        time: datetime.time,
        prev_action: Optional["ShumiAction"] = None,
        since_prev_action_duration: Optional[datetime.timedelta] = None,
        milk_type: Optional[MilkType] = None,
        milk_amount: Optional[int] = None,
        sleep_duration_min: Optional[int] = None,
        daiper_type: Optional[DaiperType] = None,
    ):
        self.action = action
        self.milk_amount = milk_amount
        self.milk_type = milk_type
        self.daiper_type = daiper_type
        self.sleep_duration_min = sleep_duration_min
        self.days = days
        date = BIRTHDAY + datetime.timedelta(days=days)
        self.date_time = datetime.datetime(
            date.year,
            date.month,
            date.day,
            time.hour,
            time.minute,
        )

        if since_prev_action_duration is not None:
            self.since_prev_action_duration = since_prev_action_duration
        elif prev_action is None:
            self.since_prev_action_duration = datetime.timedelta(0)
        elif prev_action.sleep_duration_min is None:
            self.since_prev_action_duration = self.date_time - prev_action.date_time
        else:
            prev_action_end_datetime = prev_action.date_time + datetime.timedelta(
                minutes=prev_action.sleep_duration_min
            )
            self.since_prev_action_duration = self.date_time - prev_action_end_datetime

    # String representation of the ShumiAction for easy debugging.
    def __str__(self) -> str:
        time_str = f"Day {self.days} on {self.date_time.date()}, Time {self.date_time.hour:02d}:{self.date_time.minute:02d}, {self.since_prev_action_duration} since last action"
        if self.action == Action.DRINK_MILK:
            return f"{time_str}, Action: Drink Milk({Action.DRINK_MILK.value}), Type: {self.milk_type}({self.milk_type.value if self.milk_type else 0}), Amount: {self.milk_amount} ml"
        elif self.action == Action.SLEEP:
            return f"{time_str}, Action: Sleep({Action.SLEEP.value}), Duration: {self.sleep_duration_min} minutes"
        elif self.action == Action.CHANGE_DAIPER:
            return f"{time_str}, Action: Change Daiper({Action.CHANGE_DAIPER.value}), Type: {self.daiper_type}({self.daiper_type.value if self.daiper_type else 0})"
        else:
            return "Unknown Action"


# Gets the Action enum from action string.
def getAction(action: str) -> Action:
    if action == "喝奶":
        return Action.DRINK_MILK
    elif action == "睡眠":
        return Action.SLEEP
    elif action == "换尿布":
        return Action.CHANGE_DAIPER
    else:
        raise Exception(f"Unknown Action {action}")


# Gets the MilkType enum from milk type string.
def getMilkType(milk_type: str) -> MilkType:
    if milk_type == "配方奶":
        return MilkType.BOTTLE_FORMULA_MILK
    elif milk_type == "瓶喂母乳":
        return MilkType.BOTTLE_BREAST_MILK
    elif milk_type == "亲喂母乳":
        return MilkType.BREAST_FEED
    else:
        raise Exception(f"Unknown Milk Type {milk_type}")


# Gets the DaiperType enum from daiper type string.
def getDaiperType(daiper_type: str) -> DaiperType:
    if daiper_type == "嘘嘘":
        return DaiperType.PEE
    elif daiper_type == "臭臭":
        return DaiperType.POO
    elif daiper_type == "臭臭+嘘嘘":
        return DaiperType.PEE_AND_POO
    elif daiper_type == "干爽":
        return DaiperType.CLEAN
    else:
        raise Exception(f"Unknown Daiper Type {daiper_type}")


# Gets time object from time string.
def getTime(time_str: str) -> datetime.time:
    time_parts = time_str.split(":")
    return datetime.time(
        int(time_parts[0]),
        int(time_parts[1]),
    )


# Gets datetime object from days and time.
def getDateTime(days: int, time: datetime.time) -> datetime.datetime:
    date = BIRTHDAY + datetime.timedelta(days=days)
    return datetime.datetime(
        date.year,
        date.month,
        date.day,
        time.hour,
        time.minute,
    )
