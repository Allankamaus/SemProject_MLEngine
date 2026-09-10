"""Car price-band estimator powered by an SVC classifier.

The model predicts a price band and returns the training-set median for that
band. Add more verified listings to data/cars.csv as the project grows.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


DATASET_PATH = Path(__file__).parent / "data" / "cars.csv"
NUMERIC_FEATURES = ["year", "mileage", "engine_size"]
CATEGORICAL_FEATURES = ["make", "condition", "transmission", "fuel_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
PRICE_BANDS = ("budget", "mid-range", "premium")


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
	"""Load and validate car listings from a CSV file."""
	with path.open(newline="", encoding="utf-8") as dataset_file:
		rows = list(csv.DictReader(dataset_file))

	required_columns = set(FEATURES + ["price"])
	if not rows or not required_columns.issubset(rows[0]):
		missing = required_columns.difference(rows[0] if rows else set())
		raise ValueError(f"Dataset is empty or missing columns: {sorted(missing)}")

	cleaned_rows: list[dict[str, Any]] = []
	for row in rows:
		cleaned_rows.append(
			{
				**{feature: float(row[feature]) for feature in NUMERIC_FEATURES},
				**{feature: row[feature].strip().lower() for feature in CATEGORICAL_FEATURES},
				"price": float(row["price"]),
			}
		)
	return cleaned_rows


def price_band(price: float) -> str:
	"""Map a listing price to a simple, explainable training label."""
	if price < 10000:
		return "budget"
	if price < 25000:
		return "mid-range"
	return "premium"


def train_model(rows: list[dict[str, Any]]) -> tuple[Pipeline, dict[str, float]]:
	"""Train an SVC and return representative prices for each predicted band."""
	if len(rows) < 12:
		raise ValueError("Add at least 12 listings before training the model.")

	features = pd.DataFrame([{feature: row[feature] for feature in FEATURES} for row in rows])
	labels = [price_band(row["price"]) for row in rows]
	if len(set(labels)) < 3:
		raise ValueError("The dataset must contain examples from all three price bands.")

	preprocessor = ColumnTransformer(
		transformers=[
			("numeric", StandardScaler(), NUMERIC_FEATURES),
			("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
		]
	)
	model = Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced")),
		]
	)
	model.fit(features, labels)

	representative_prices = {
		band: sum(row["price"] for row in rows if price_band(row["price"]) == band)
		/ sum(price_band(row["price"]) == band for row in rows)
		for band in PRICE_BANDS
	}
	return model, representative_prices


def estimate_price(
	model: Pipeline,
	representative_prices: dict[str, float],
	car: dict[str, Any],
) -> dict[str, Any]:
	"""Estimate a car's price band, representative price, and confidence."""
	model_input = pd.DataFrame([{feature: car[feature] for feature in FEATURES}])
	band = str(model.predict(model_input)[0])
	probabilities = model.predict_proba(model_input)[0]
	confidence = float(max(probabilities))
	return {
		"price_band": band,
		"estimated_price": round(representative_prices[band], 2),
		"confidence": round(confidence, 3),
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Estimate a used car's price band.")
	parser.add_argument("--year", type=float, required=True)
	parser.add_argument("--make", required=True)
	parser.add_argument("--mileage", type=float, required=True)
	parser.add_argument("--condition", choices=("poor", "fair", "good", "excellent"), required=True)
	parser.add_argument("--engine-size", type=float, required=True)
	parser.add_argument("--transmission", choices=("manual", "automatic"), required=True)
	parser.add_argument("--fuel-type", choices=("petrol", "diesel", "hybrid", "electric"), required=True)
	args = parser.parse_args()

	model, representative_prices = train_model(load_dataset())
	result = estimate_price(model, representative_prices, vars(args) | {"engine_size": args.engine_size})
	print(f"Estimated price: ${result['estimated_price']:,.2f}")
	print(f"Price band: {result['price_band']}")
	print(f"Model confidence: {result['confidence']:.1%}")
	print("Use this as a guide; inspect the car and compare current local listings before buying.")


if __name__ == "__main__":
	main()
