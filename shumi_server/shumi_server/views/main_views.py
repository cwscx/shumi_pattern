# The functions for main page
import json
import os
import re
from datetime import datetime
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse

JSON_FILE_PATH = os.path.join(settings.BASE_DIR, "shumi_server", "shumi.json")


def index_view(request):
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


def analytics(request):
    json_path = os.path.join(settings.BASE_DIR, "shumi_server", "shumi.json")
    
    # Default empty data if file missing
    db = {"patterns": [], "info": {}}
    
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            db = json.load(file)

    # Pass the full patterns list so JS can calculate the rolling 24h totals
    context = {
        "patterns_json": json.dumps(db.get("patterns", [])),
        "info": db.get("info", {}),
        "active_page": "analytics" 
    }
    return render(request, "analytics.html", context)