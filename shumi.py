from google import genai
import json

GEMINI_API_KEY = "AIzaSyBamJpJPhaWweT0AGiuZ_104avUerAhSOc"
client = genai.Client(api_key=GEMINI_API_KEY)

# Open the file in read mode ('r')
with open("shumi.json", "r") as file:
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


patterns = shumi_pattern["patterns"]
milk_patterns = getPatterns("喝奶")
daiper_patterns = getPatterns("换尿布")
sleep_patterns = getPatterns("睡眠")

user_query = input(
    "Ask your question (or Press Enter to see the default analysis): "
).strip()

prompts = [
    # user query
    f"User's initial query {user_query}.",
    # context
    f"Here's the behavior pattern of my daughter 施舒米 {shumi_pattern}.",
    "----------",
    # role-specific prompt
    "You are an infant behavior prediction assistant which offers emotional support for parents.",
    # COT
    "If there is a reasoning process to generate the response, think step by step and put your steps in bullet points.",
    (
        f"""
        Please do the following steps:
        1. Predict her next possible actions;
        2. Summarize her actions in the last 3 days in a clear and succinct way;
        3. Based on {milk_patterns}, analyze her long-term milk drinking behavior;
        4. Based on {daiper_patterns}, analyzer her long-term daiper behavior;
        5. Based on {sleep_patterns}, analyze her long-term sleep behavior;
        """
        if len(user_query) == 0
        else "Please answer user's initial query;"
    ),
    "----------",
    # user context prompt.
    f"""
    Use the following user profile to personalize the output.
    Write in Chinese. If a day hasn't finished yet, only use that date's data for prediction, but not for summarization.
    """,
]

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="\n".join(prompts)
)
print(response.text)
