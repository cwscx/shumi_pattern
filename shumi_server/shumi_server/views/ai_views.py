# AI related functions
import json
import os
import re
from google import genai
from datetime import datetime
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from inference import predict_next_actions


# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
JSON_FILE_PATH = os.path.join(settings.BASE_DIR, "shumi_server", "shumi.json")


def getTime(timeStr):
    time_parts = timeStr.split(":")
    now = datetime.now()
    return datetime(
        now.year, now.month, now.day, int(time_parts[0]), int(time_parts[1])
    )


def isWithinTimeWindow(timeWindowHour, time1, time2):
    time_difference = min(abs(time1 - time2), abs(time2 - time1))
    return abs(time_difference.seconds) <= (timeWindowHour * 60 * 60)


@csrf_exempt
def get_gemini_insights(request):
    if request.method != "POST":
        return JsonResponse(
            {"status": "error", "message": "Only POST allowed"}, status=405
        )

    try:
        # 1. Load Data
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
            shumi_pattern = json.load(file)

        basic_info = shumi_pattern.get("info", {})
        patterns = shumi_pattern.get("patterns", [])

        # 2. Extract specific patterns for the AI
        def get_specific_actions(action_type):
            return [
                {
                    "date": day["date"],
                    "actions": [
                        a
                        for a in day.get("actions", [])
                        if a.get("action") == action_type
                    ],
                }
                for day in patterns
            ]

        milk_data = get_specific_actions("喝奶")
        diaper_data = get_specific_actions("换尿布")
        sleep_data = get_specific_actions("睡眠")

        # 3. Parse User Query
        body = json.loads(request.body) if request.body else {}
        user_query = body.get("query", "").strip()

        # 4. Construct the Structured Prompt
        # System Instructions
        system_prompt = """
        Role: 你是一位资深婴儿行为专家和情感支持助手。
        Context: 这是我女儿的生日是2025/9/6
        Instructions:
        - 必须使用中文回答。
        - 如果用户提出了具体问题，请**优先且直接**回答该问题。
        - 在回答之后，提供基于数据的深度分析。
        - 推理过程请使用分点（Bullet Points）。
        """

        behavioral_logic = """
		<DATA_INTERPRETATION_RULES>
		1. 晚间睡眠识别：从每天最后一觉到次日第一觉，若中间清醒时间极短（如仅为了喂奶/换尿布且处于半梦半醒状态），必须将其视为一个【完整的晚间睡眠】，而非多次小睡。
		2. 碎片化小睡处理：如果两个小觉（Nap）之间间隔小于1小时且没有喝奶记录，这属于“接觉失败”或“断断续续的睡眠”，应合并为一个【断断续续的长觉】，中间的间隔不计为有效的“清醒间隔”。
		3. 进行中状态识别：如果最近的一条睡眠记录没有“结束时间”，表示舒米【正在睡觉】，请基于此状态进行后续预判。
		</DATA_INTERPRETATION_RULES>
		"""

        # Data Blocks using XML-style tags for clarity
        data_context = f"""
        <BABY_INFO> {json.dumps(basic_info, ensure_ascii=False)} </BABY_INFO>
        <MILK_LOGS> {json.dumps(milk_data[-7:], ensure_ascii=False)} </MILK_LOGS>
        <SLEEP_LOGS> {json.dumps(sleep_data[-7:], ensure_ascii=False)} </SLEEP_LOGS>
        <DIAPER_LOGS> {json.dumps(diaper_data[-7:], ensure_ascii=False)} </DIAPER_LOGS>
        <FULL_HISTORY> {json.dumps(patterns[-3:], ensure_ascii=False)} </FULL_HISTORY>
        """

        # Task Logic
        if not user_query:
            task_prompt = """
            User has not asked a specific question. Please provide a general report:
            1. 预测接下来的动作及时间区间（含置信度）。请与本地模型的预测比较并总结。
            2. 简要总结过去3天的整体表现,以及是否符合当前月龄。
            3. 分析长期的喂奶、睡眠和尿布规律。
            """
        else:
            task_prompt = f"""
            重要任务：用户提出了一个具体问题，请结合上述所有数据，给出专业且详尽的解答。
            用户问题："{user_query}"
            """

        # Final Assembly: Instructions -> Data -> Specific Task
        final_prompt = f"{system_prompt}\n{behavioral_logic}\n{data_context}\n{task_prompt}"

        # 5. Call Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # Highly recommend 2.0-flash for speed/logic
            contents=final_prompt,
        )

        return JsonResponse({"status": "success", "insights": response.text})

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@csrf_exempt
def get_prediction(request):
    # Case 1: Handle if the request is NOT a POST
    if request.method != "POST":
        return JsonResponse(
            {"status": "error", "message": "Invalid request method"}, status=400
        )

    next_actions = predict_next_actions(50)
    predictions = ""
    for next_action in next_actions:
        prob = next_action[1]["action_type"][next_action[0].action.value]
        predictions += f"{next_action[0]} with probablity {prob * 100:.2f}%\n"
    print(predictions)

    try:
        # Load the file
        if not os.path.exists(JSON_FILE_PATH):
            return JsonResponse(
                {"status": "error", "message": "Data file not found"}, status=404
            )

        with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
            db = json.load(file)

        basic_info = db.get("info", {})
        patterns = db.get("patterns", [])
        current_time = datetime.now()

        prompt = f"""
        基于施舒米的信息 {basic_info} 和作息 {patterns}。
        当前时间 {current_time}，如果作息记录严重缺失，请混合使用本地模型推测的数据来推断：{predictions}。
        
        请预测下一个动作，并给出你对此预测的信心指数（0-100%）。
        输出格式：
        CONFIDENCE: [数字]
        PREDICTION: [动作] [时间]
        REASONING: [推理]
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        full_text = response.text

        # Default values
        confidence = 70
        prediction = "信号不准"
        reasoning = "无法获取推理"

        # Parsing
        if "CONFIDENCE:" in full_text:
            conf_match = re.search(r"CONFIDENCE:\s*(\d+)", full_text)
            if conf_match:
                confidence = conf_match.group(1)

        if "PREDICTION:" in full_text:
            prediction = (
                full_text.split("PREDICTION:")[1].split("REASONING:")[0].strip()
            )

        if "REASONING:" in full_text:
            reasoning = full_text.split("REASONING:")[1].strip()

        # Case 2: Successful return
        return JsonResponse(
            {
                "status": "success",
                "prediction": prediction,
                "confidence": confidence,
                "reasoning": reasoning,
            }
        )

    except Exception as e:
        # Case 3: Error return (Crucial! If you skip this, it returns None)
        return JsonResponse({"status": "error", "message": str(e)}, status=500)