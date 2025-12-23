from google import genai
import json

GEMINI_API_KEY = "AIzaSyCezyEraOsEUjD7jV9CowQcGsOLO-3qbgE"
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

prompt = f"""
你是一个婴儿行为预测助手。这里是我的女儿施舒米的每天干的事{shumi_pattern}。请帮我依次做如下事情：
1. 预测她下一次的行为最有可能是什么以及什么时候发生。
2. 总结过去3天她的行为是否符合她这个年龄段的宝宝。
3. 基于{milk_patterns}分析她的长期喝奶行为
4. 基于{daiper_patterns}分析她的长期换尿布行为
5. 基于{sleep_patterns}分析她的长期睡眠行为
"""

response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
print(response.text)
