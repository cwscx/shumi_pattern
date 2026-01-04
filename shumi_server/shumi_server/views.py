import json
import os
import re
from google import genai
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from inference import predict_next_actions

# Path to your JSON file
JSON_FILE_PATH = os.path.join(os.path.dirname(__file__), "shumi.json")


def index_view(request):
    # Get the current date in PST for the "Default Date" input
    # If USE_TZ = False and TIME_ZONE is set, this returns PST
    current_pst_date = datetime.now().strftime("%Y-%m-%d")
    past_days_data = []
    if os.path.exists(JSON_FILE_PATH):
        try:
            with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
                db = json.load(f)
                all_patterns = db.get("patterns", [])
                all_patterns.sort(key=lambda x: x["date"], reverse=True)
                past_days_data = all_patterns[:5]

                for day in past_days_data:
                    for i, action in enumerate(day["actions"]):
                        action["index_in_file"] = i

                        # Calculate Sleep Duration
                        if (
                            action["action"] == "睡眠"
                            and action.get("time_start")
                            and action.get("time_end")
                        ):
                            try:
                                fmt = "%H:%M"
                                t1 = datetime.strptime(action["time_start"], fmt)
                                t2 = datetime.strptime(action["time_end"], fmt)
                                # Handle sleep passing midnight
                                delta = t2 - t1
                                if delta.days < 0:
                                    tdelta_seconds = delta.seconds
                                else:
                                    tdelta_seconds = delta.total_seconds()

                                hours = int(tdelta_seconds // 3600)
                                minutes = int((tdelta_seconds % 3600) // 60)
                                action["duration"] = f"{hours}h {minutes}m"
                            except:
                                action["duration"] = "时间错误"

                    day["actions"].sort(key=lambda x: x["time_start"], reverse=True)
        except:
            past_days_data = []

    # Prepare Chart Data
    chart_data = {"milk": [], "sleep": [], "diaper": []}

    for day in past_days_data:
        date_str = day["date"]
        for action in day["actions"]:
            try:
                # Convert "HH:MM" to decimal hours
                h, m = map(int, action["time_start"].split(":"))
                start = round(h + (m / 60), 2)

                # Determine block end
                if action["action"] == "睡眠" and action.get("time_end"):
                    h_e, m_e = map(int, action["time_end"].split(":"))
                    end = round(h_e + (m_e / 60), 2)

                    # Handle midnight crossing (split the block)
                    if end < start:
                        # Block 1: From start to midnight
                        chart_data["sleep"].append({"x": date_str, "y": [start, 24]})
                        # Note: The 'next day' portion is usually covered by its own entry
                        # or you can manually inject it if your data structure differs.
                        continue
                elif action["action"] == "喝奶":
                    end = start + 0.4  # 24 minute block for visibility
                else:
                    end = start + 0.2  # 12 minute block for diapers

                # Assign to category
                key = (
                    "milk"
                    if action["action"] == "喝奶"
                    else "sleep" if action["action"] == "睡眠" else "diaper"
                )
                chart_data[key].append(
                    {
                        "x": date_str,
                        "y": [start, end],
                        "label": action["action"],
                        "details": f"{action.get('type', '')} {action.get('volume', '')}".strip(),
                    }
                )
            except:
                continue

    return render(
        request,
        "index.html",
        {
            "past_days_data": past_days_data,
            "chart_data_json": json.dumps(chart_data),
            "current_pst_date": current_pst_date,
        },
    )


@csrf_exempt
def save_baby_data(request):
    if request.method == "POST":
        try:
            new_data = json.loads(request.body)

            # 1. Load existing data or create initial structure
            if os.path.exists(JSON_FILE_PATH):
                with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
                    db = json.load(f)
            else:
                db = {
                    "name": "施舒米",
                    "birthday": "2025/09/06",
                    "sex": "female",
                    "patterns": [],
                }

            # 2. Update logic: Find the date in patterns or add new
            target_date = new_data["date"]
            day_entry = next(
                (p for p in db["patterns"] if p["date"] == target_date), None
            )

            if day_entry:
                day_entry["actions"].append(new_data["action_item"])
            else:
                db["patterns"].append(
                    {"date": target_date, "actions": [new_data["action_item"]]}
                )

            # 3. Save back to file
            with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(db, f, ensure_ascii=False, indent=4)

            return JsonResponse({"status": "success"})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)


@csrf_exempt
def delete_baby_action(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            target_date = data.get("date")
            target_index = data.get("index")

            if os.path.exists(JSON_FILE_PATH):
                with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
                    db = json.load(f)

                # Find the correct day
                for day_entry in db.get("patterns", []):
                    if day_entry["date"] == target_date:
                        # Remove the item at the specific index
                        if 0 <= target_index < len(day_entry["actions"]):
                            day_entry["actions"].pop(target_index)
                            break

                # Save the updated data
                with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
                    json.dump(db, f, ensure_ascii=False, indent=4)

                return JsonResponse({"status": "success"})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    return JsonResponse({"status": "invalid method"}, status=405)


############################## Ai insights related functions ##############################

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
JSON_FILE_PATH = os.path.join(os.path.dirname(__file__), "shumi.json")


def getTime(timeStr):
    time_parts = timeStr.split(":")
    now = datetime.now()
    return datetime(
        now.year, now.month, now.day, int(time_parts[0]), int(time_parts[1])
    )


def isWithinTimeWindow(timeWindowHour, time1, time2):
    time_difference = min(abs(time1 - time2), abs(time2 - time1))
    return abs(time_difference.seconds) <= (timeWindowHour * 60 * 60)


# @csrf_exempt
# def get_gemini_insights(request):
#     if request.method != 'POST':
#         return JsonResponse({"error": "Only POST allowed"}, status=405)

#     try:
#         # Load the latest data from shumi.json
#         with open(JSON_FILE_PATH, "r", encoding='utf-8') as file:
#             shumi_pattern = json.load(file)

#         basic_info = shumi_pattern.get("info", {})
#         patterns = shumi_pattern.get("patterns", [])

#         # Logic to extract sub-patterns (from your script)
#         def getPatterns(action_type):
#             return [{"date": day["date"], "actions": [a for a in day.get("actions", []) if a.get("action") == action_type]} for day in patterns]

#         def getTimePatterns():
#             now = datetime.now()
#             return [{"date": day["date"], "actions": [a for a in day.get("actions", []) if isWithinTimeWindow(2, getTime(a.get("time_start")), now)]} for day in patterns]

#         milk_patterns = getPatterns("喝奶")
#         daiper_patterns = getPatterns("换尿布")
#         sleep_patterns = getPatterns("睡眠")
#         time_patterns = getTimePatterns()

#         # Get user query from frontend if it exists
#         body = json.loads(request.body) if request.body else {}
#         user_query = body.get("query", "").strip()

#         print("******* user query is ")
#         print(user_query)

#         # Build Prompts (Your CoT Logic)
#         prompts = [
#            f"Here's the basic info of my daughter 施舒米 {basic_info}",
#     "----------",
#     # role-specific prompt
#     "You are an infant behavior prediction assistant which offers emotional support for parents.",
#     # COT
#     "If there is a reasoning process to generate the response, think step by step and put your steps in bullet points. ",
#     "For example, when you predict, you should calculate the time difference between each actions instead of just predicting from the previous timestamp.",
#     # user query.
#     (
#         f"""
#         Please do the following steps:
#         1. Based on {time_patterns}, predict her next possible actions and time ranges with confidence interval;
#         2. Summarize her actions in the last 3 days in a clear and succinct way;
#         3. Based on {milk_patterns}, analyze her long-term milk drinking behavior;
#         4. Based on {daiper_patterns}, analyzer her long-term daiper behavior;
#         5. Based on {sleep_patterns}, analyze her long-term sleep behavior;
#         """
#         if len(user_query) == 0
#         else f"Based on 施舒米's behavior patterns {patterns}, please answer user's initial query;"
#     ),
#     "----------",
#     # user context prompt.
#     f"""
#     Use the following user profile to personalize the output.
#     Write in Chinese. If a day hasn't finished yet, only use that date's data for prediction, but not for summarization.
#     The default timezone is PST, but if the timezone changes, please consider jet lag impact when analyzing, and provide suggestioins accordingly.
#     """,
#         ]

#         # Call Gemini (Non-streaming for simpler UI display)
#         response = client.models.generate_content(
#             model="gemini-2.5-flash", contents="\n".join(prompts)
#         )

#         return JsonResponse({
#             "status": "success",
#             "insights": response.text
#         })

#     except Exception as e:
#         return JsonResponse({"status": "error", "message": str(e)}, status=500)


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
            2. 简要总结过去3天的整体表现。
            3. 分析长期的喂奶、睡眠和尿布规律。
            """
        else:
            task_prompt = f"""
            重要任务：用户提出了一个具体问题，请结合上述所有数据，给出专业且详尽的解答。
            用户问题："{user_query}"
            """

        # Final Assembly: Instructions -> Data -> Specific Task
        final_prompt = f"{system_prompt}\n{data_context}\n{task_prompt}"

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
        当前时间 {current_time}，如果作息记录严重缺失，请使用本地模型推测的数据来推断：{patterns}。
        
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
