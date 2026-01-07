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
    if request.method != "POST":
        return JsonResponse(
            {"status": "error", "message": "Only POST allowed"}, status=405
        )

    try:
        # 2. Parse request body
        body = json.loads(request.body) if request.body else {}
        use_local = body.get("useLocalModel", False)

        # 3. Handle Local Model Logic
        local_model_output = ""
        if use_local:
            try:
                # Assuming predict_next_actions is imported/defined
                next_actions = predict_next_actions(50)
                for next_action in next_actions:
                    # Accessing the action and probability
                    action_obj = next_action[0]
                    prob = next_action[1]["action_type"][action_obj.action.value]
                    local_model_output += f"{action_obj} with probability {prob * 100:.2f}%\n"
            except Exception as e:
                local_model_output = f"Error running local model: {str(e)}"

        # 4. Resolve JSON Path using Django Settings
        # This fixes the "file not found" error in modular views
        json_path = os.path.join(settings.BASE_DIR, "shumi_server", "shumi.json")
        
        if not os.path.exists(json_path):
            return JsonResponse(
                {"status": "error", "message": f"Data file not found at {json_path}"}, status=404
            )

        with open(json_path, "r", encoding="utf-8") as file:
            db = json.load(file)

        basic_info = db.get("info", {})
        # Send only the last few days to Gemini to avoid context overload
        recent_patterns = db.get("patterns", [])[-5:] 
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 5. Construct Prompt (Fixed variable names)
        prompt = f"""
        你是一位资深育儿专家。请根据以下数据进行预判。
        
        [系统状态]
        本地模型内核: {'开启' if use_local else '关闭'}
        本地模型原始输出: {local_model_output if local_model_output else '无'}
        当前时间: {current_time} (PST)

        [背景数据]
        舒米信息: {basic_info}
        最近作息记录: {recent_patterns}

        [分析要求]
        1. 如果作息记录缺失，请优先参考“本地模型原始输出”。
        2. 遵守以下睡眠逻辑：清醒时间小于1小时的小觉应视为断断续续的同一个觉；凌晨为了喂奶换尿布的清醒不算作正式起夜。
        3. 给出准确的信心指数。

        [输出格式]
        CONFIDENCE: [0-100的数字]
        PREDICTION: [动作] [预判时间点]
        REASONING: [简洁的中文推理]
        """

        # 6. Call Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        full_text = response.text

        # 7. Robust Parsing
        confidence = 70
        prediction = "信号捕捉中"
        reasoning = "推理生成中"

        # Extract Confidence
        conf_match = re.search(r"CONFIDENCE:\s*(\d+)", full_text)
        if conf_match:
            confidence = int(conf_match.group(1))

        # Extract Prediction
        if "PREDICTION:" in full_text:
            parts = full_text.split("PREDICTION:")[1].split("REASONING:")
            prediction = parts[0].strip()
            if len(parts) > 1:
                reasoning = parts[1].strip()

        return JsonResponse({
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "reasoning": reasoning,
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc()) # Print the error to your terminal
        return JsonResponse({"status": "error", "message": f"Server Error: {str(e)}"}, status=500)