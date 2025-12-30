#!/bin/bash
python3 -m pip list --outdated  | cut -d '=' -f 1 | xargs -n1 | awk 'NR % 4 == 1' |  awk 'NR > 2' |  xargs -n1 python3 -m pip install -U

python3 -m pip install google
python3 -m pip install google-api-python-client 
python3 -m pip install google-genai
python3 -m pip install torch
