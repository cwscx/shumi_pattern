# APIs to handle data
import json
import os
import re
from datetime import datetime
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


# Path to your JSON file
#JSON_FILE_PATH = os.path.join(os.path.dirname(__file__), "shumi.json")
JSON_FILE_PATH = os.path.join(settings.BASE_DIR, "shumi_server", "shumi.json")

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