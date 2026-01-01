# shumi_pattern
LLM agent to analyze Shumi's baby pattern.

## Gemini Script

```
python3 -W ignore  shumi.py
```

## Web Server

```
python3 shumi_server/manage.py runserver
```

## Model Training 

Running the following command will retrain the model with the latest data and save it.

```
python3 shumi_model/train.py
```

## Model Predict

```
python3 shumi_model/predict.py
```