import json
import os
from google import genai
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Path to your JSON file
JSON_FILE_PATH = os.path.join(os.path.dirname(__file__), 'shumi.json')

def index_view(request):
    past_days_data = []
    
    if os.path.exists(JSON_FILE_PATH):
        try:
            with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
                db = json.load(f)                
                all_patterns = db.get('patterns', [])                
                # Sort and slice
                all_patterns.sort(key=lambda x: x['date'], reverse=True)
                past_days_data = all_patterns[:5]
        except Exception as e:
            print(f"DEBUG: Error reading JSON: {e}")
    else:
        print("DEBUG: File does not exist at the specified path!")
            
    return render(request, 'index.html', {'past_days_data': past_days_data})


@csrf_exempt
def save_baby_data(request):
    if request.method == 'POST':
        try:
            new_data = json.loads(request.body)
            
            # 1. Load existing data or create initial structure
            if os.path.exists(JSON_FILE_PATH):
                with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
                    db = json.load(f)
            else:
                db = {
                    "name": "施舒米",
                    "birthday": "2025/09/06",
                    "sex": "female",
                    "patterns": []
                }

            # 2. Update logic: Find the date in patterns or add new
            target_date = new_data['date']
            day_entry = next((p for p in db['patterns'] if p['date'] == target_date), None)
            
            if day_entry:
                day_entry['actions'].append(new_data['action_item'])
            else:
                db['patterns'].append({
                    "date": target_date,
                    "actions": [new_data['action_item']]
                })

            # 3. Save back to file
            with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(db, f, ensure_ascii=False, indent=4)

            return JsonResponse({"status": "success"})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)

@csrf_exempt
def delete_baby_action(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            target_date = data.get('date')
            target_index = data.get('index')

            if os.path.exists(JSON_FILE_PATH):
                with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
                    db = json.load(f)
                
                # Find the correct day
                for day_entry in db.get('patterns', []):
                    if day_entry['date'] == target_date:
                        # Remove the item at the specific index
                        if 0 <= target_index < len(day_entry['actions']):
                            day_entry['actions'].pop(target_index)
                            break
                
                # Save the updated data
                with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
                    json.dump(db, f, ensure_ascii=False, indent=4)
                
                return JsonResponse({"status": "success"})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    return JsonResponse({"status": "invalid method"}, status=405)

############################## Ai insights related functions ##############################

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
JSON_FILE_PATH = os.path.join(os.path.dirname(__file__), 'shumi.json')

def getTime(timeStr):
    time_parts = timeStr.split(":")
    now = datetime.now()
    return datetime(now.year, now.month, now.day, int(time_parts[0]), int(time_parts[1]))

def isWithinTimeWindow(timeWindowHour, time1, time2):
    time_difference = min(abs(time1 - time2), abs(time2 - time1))
    return abs(time_difference.seconds) <= (timeWindowHour * 60 * 60)

@csrf_exempt
def get_gemini_insights(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        # Load the latest data from shumi.json
        with open(JSON_FILE_PATH, "r", encoding='utf-8') as file:
            shumi_pattern = json.load(file)
        
        basic_info = shumi_pattern.get("info", {})
        patterns = shumi_pattern.get("patterns", [])

        # Logic to extract sub-patterns (from your script)
        def getPatterns(action_type):
            return [{"date": day["date"], "actions": [a for a in day.get("actions", []) if a.get("action") == action_type]} for day in patterns]

        def getTimePatterns():
            now = datetime.now()
            return [{"date": day["date"], "actions": [a for a in day.get("actions", []) if isWithinTimeWindow(2, getTime(a.get("time_start")), now)]} for day in patterns]

        milk_patterns = getPatterns("喝奶")
        daiper_patterns = getPatterns("换尿布")
        sleep_patterns = getPatterns("睡眠")
        time_patterns = getTimePatterns()

        # Get user query from frontend if it exists
        body = json.loads(request.body) if request.body else {}
        user_query = body.get("query", "").strip()

        # Build Prompts (Your CoT Logic)
        prompts = [
            f"Here's the basic info of my daughter 施舒米 {basic_info}",
            "You are an infant behavior prediction assistant which offers emotional support for parents.",
            "If there is a reasoning process, think step by step in bullet points. Calculate time differences for predictions.",
            (
                f"Steps: 1. Predict next actions based on {time_patterns}; 2. Summarize last 3 days; 3. Analyze long-term milk {milk_patterns}; 4. Analyze diapers {daiper_patterns}; 5. Analyze sleep {sleep_patterns}."
                if not user_query else f"Based on patterns {patterns}, answer: {user_query}"
            ),
            "Use the profile to personalize. Write in Chinese. Use Markdown formatting for clarity."
        ]

        # Call Gemini (Non-streaming for simpler UI display)
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents="\n".join(prompts)
        )

        return JsonResponse({
            "status": "success",
            "insights": response.text
        })

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
