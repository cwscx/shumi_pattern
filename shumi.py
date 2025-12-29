from google import genai
import json
import time
import datetime
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Open the file in read mode ('r')
with open("shumi_server/shumi_server/shumi.json", "r") as file:
    # Use json.load() to deserialize the file object into a Python dictionary
    shumi_pattern = json.load(file)


# Extracts the patterns for a specific action type.
def getPatterns(action_type: str = "喝奶"):
    return [
        {
            "date": day["date"],
            "actions": [
                action
                for action in day.get("actions", [])
                if action.get("action") == action_type
            ],
        }
        for day in patterns
    ]


def getTime(timeStr: str) -> datetime.datetime:
    time_parts = timeStr.split(":")
    now = datetime.datetime.now()
    return datetime.datetime(
        now.year,
        now.month,
        now.day,
        int(time_parts[0]),
        int(time_parts[1]),
    )


def isWithinTimeWindow(
    timeWindowHour: int, time1: datetime.datetime, time2: datetime.datetime
) -> bool:
    time_difference = min(abs(time1 - time2), abs(time2 - time1))
    return abs(time_difference.seconds) <= (timeWindowHour * 60 * 60)


def getTimePatterns():
    return [
        {
            "date": day["date"],
            "actions": [
                action
                for action in day.get("actions", [])
                if isWithinTimeWindow(
                    2, getTime(action.get("time_start")), datetime.datetime.now()
                )
            ],
        }
        for day in patterns
    ]


basic_info = {
    "name": shumi_pattern["name"],
    "birthday": shumi_pattern["birthday"],
    "sex": shumi_pattern["sex"],
}
patterns = shumi_pattern["patterns"]
milk_patterns = getPatterns("喝奶")
daiper_patterns = getPatterns("换尿布")
sleep_patterns = getPatterns("睡眠")
time_patterns = getTimePatterns()

user_query = input(
    "Ask your question (or Press Enter to see the default analysis): "
).strip()

prompts = [
    # context
    f"Here's the basic info of my daughter 施舒米 {basic_info}",
    "----------",
    # role-specific prompt
    "You are an infant behavior prediction assistant which offers emotional support for parents.",
    # COT
    "If there is a reasoning process to generate the response, think step by step and put your steps in bullet points. ",
    "For example, when you predict, you should calculate the time difference between each actions instead of just predicting from the previous timestamp.",
    # user query.
    (
        f"""
        Please do the following steps:
        1. Based on {time_patterns}, predict her next possible actions and time ranges with confidence interval;
        2. Summarize her actions in the last 3 days in a clear and succinct way;
        3. Based on {milk_patterns}, analyze her long-term milk drinking behavior;
        4. Based on {daiper_patterns}, analyzer her long-term daiper behavior;
        5. Based on {sleep_patterns}, analyze her long-term sleep behavior;
        """
        if len(user_query) == 0
        else f"Based on 施舒米's behavior patterns {patterns}, please answer user's initial query;"
    ),
    "----------",
    # user context prompt.
    f"""
    Use the following user profile to personalize the output.
    Write in Chinese. If a day hasn't finished yet, only use that date's data for prediction, but not for summarization.
    """,
]

start_time = time.time()

stream = client.models.generate_content_stream(
    model="gemini-2.5-flash", contents="\n".join(prompts)
)
out = []
for chunk in stream:
    if getattr(chunk, "text", None):
        print(chunk.text, end="", flush=True)
        out.append(chunk.text)

print()
print(f"[latency] total seconds: {time.time() - start_time:.2f}")
