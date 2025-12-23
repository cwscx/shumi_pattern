from google import genai
import json

GEMINI_API_KEY = "AIzaSyCezyEraOsEUjD7jV9CowQcGsOLO-3qbgE"
client = genai.Client(api_key=GEMINI_API_KEY)

# Open the file in read mode ('r')
with open("shumi.json", "r") as file:
    # Use json.load() to deserialize the file object into a Python dictionary
    shumi_pattern = json.load(file)

prompt = f"""
你是一个婴儿行为预测助手。这里是我的女儿施舒米的每天干的事{shumi_pattern}。请帮我预测她下一次的行为最有可能是什么以及什么时候发生。
并且总结今天她的行为是否符合她这个年龄段的宝宝。
"""

response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
print(response.text)
