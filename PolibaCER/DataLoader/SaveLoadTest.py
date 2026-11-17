from sklearn.metrics import classification_report

from PolibaCER.DataLoader.DataLoader import save_prediction_to_json, load_predictions_from_json

def calculate_metrics(results):
    labels_order = ["SUPPORT", "CONTRADICT", "NEI"]

    label_mapping = {
        "TRUE": "SUPPORT",
        "SUPPORTED": "SUPPORT",
        "SUPPORT": "SUPPORT",
        "FALSE": "CONTRADICT",
        "CONTRADICTED": "CONTRADICT",
        "CONTRADICT": "CONTRADICT",
        "NEI": "NEI",
        "NOT ENOUGH INFORMATION": "NEI",
    }

    y_true = []
    y_pred = []

    for item in results.values():
        true_label = str(item["true_label"]).strip().upper()
        predicted_label = str(item["predicted_label"]).strip().upper()

        if true_label not in label_mapping:
            raise ValueError(f"True label non riconosciuta: {true_label}")

        if predicted_label not in label_mapping:
            raise ValueError(f"Predicted label non riconosciuta: {predicted_label}")

        y_true.append(label_mapping[true_label])
        y_pred.append(label_mapping[predicted_label])

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_order,
        target_names=labels_order,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    return report

saved_predictions = load_predictions_from_json("TestMatrixCalc.json")

metrics = calculate_metrics(saved_predictions)

print("\nMetriche per classe:")
for label in ["SUPPORT", "CONTRADICT", "NEI"]:
    print(
        f"{label}: "
        f"precision={metrics[label]['precision']:.4f}, "
        f"recall={metrics[label]['recall']:.4f}, "
        f"F1={metrics[label]['f1-score']:.4f}"
    )

print("\nMetriche aggregate:")
print(f"Macro precision: {metrics['macro avg']['precision']:.4f}")
print(f"Macro recall:    {metrics['macro avg']['recall']:.4f}")
print(f"Macro F1:        {metrics['macro avg']['f1-score']:.4f}")
print(f"Accuracy:        {metrics['accuracy']:.4f}")




