from enum import Enum
from typing import Optional


class Action(Enum):
    DRINK_MILK = 1
    SLEEP = 2
    CHANGE_DAIPER = 3


class MilkType(Enum):
    BOTTLE_BREAST_MILK = 1
    BOTTLE_FORMULA_MILK = 2
    BREAST_FEED = 3


class DaiperType(Enum):
    PEE = 1
    POO = 2
    PEE_AND_POO = 3
    CLEAN = 4


class ShumiAction:
    day: int
    day_time_hour: int
    day_time_minute: int
    action: Action
    milk_type: Optional[MilkType] = None
    milk_amount: Optional[float] = None
    sleep_duration: Optional[int] = None
    daiper_type: Optional[DaiperType] = None

    def __init__(
        self,
        action: Action,
        day: int,
        day_time_hour: int,
        day_time_minute: int,
        milk_type: Optional[MilkType] = None,
        milk_amount: Optional[float] = None,
        sleep_duration: Optional[int] = None,
        daiper_type: Optional[DaiperType] = None,
    ):
        self.action = action
        self.MilkType = MilkType
        self.milk_amount = milk_amount
        self.sleep_duration = sleep_duration
        self.DaiperType = DaiperType
        self.day = day
        self.day_time_hour = day_time_hour
        self.day_time_minute = day_time_minute


def getAction(action: str) -> Action:
    if action == "喝奶":
        return Action.DRINK_MILK
    elif action == "睡眠":
        return Action.SLEEP
    elif action == "换尿布":
        return Action.CHANGE_DAIPER
    else:
        raise Exception("Unknown Action")


def getMilkType(milk_type: str) -> MilkType:
    if milk_type == "配方奶":
        return MilkType.BOTTLE_FORMULA_MILK
    elif milk_type == "瓶喂母乳":
        return MilkType.BOTTLE_BREAST_MILK
    elif milk_type == "亲喂母乳":
        return MilkType.BREAST_FEED
    else:
        raise Exception("Unknown Milk Type")


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
        raise Exception("Unknown Daiper Type")
