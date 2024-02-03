1. Build the models by running the main file in projects/ml_models
2. Change directory to project
3. Run uvicorn main:app --reload to fire up the fastapi server
4. Requests can be sent like this: http://127.0.0.1:8000/check_label/?text={someText}
