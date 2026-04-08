from src.algorithm_engine.predict import predict_with_model

preds = predict_with_model(
    "sarimax",
    "2025-01-01",
    "2025-01-07"
)

print(preds)
