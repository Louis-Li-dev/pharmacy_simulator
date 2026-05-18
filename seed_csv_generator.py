import argparse
import csv
import math
import random
from pathlib import Path


OUTPUT_COLUMNS = [
    "client_id",
    "age",
    "disease",
    "doctor_visit_day",
    "pharmacy_report_day",
    "m_days",
    "meds_received",
    "retail_products",
]


def split_items(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def read_config(path: Path) -> dict:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    params = {
        "n_clients": 200,
        "simulation_days": 2880,
        "split_ratio": 0.8,
        "seq_len": 5,
        "seed": 42,
        "max_basket_items": 4,
    }
    disease_meds = {}
    pathways = {}
    products = []
    modifiers = {}

    for row in rows:
        record_type = row.get("record_type", "").strip()
        if record_type == "parameter":
            name = row.get("name", "").strip()
            value = row.get("value", "").strip()
            if name:
                params[name] = float(value) if "." in value else int(value)
        elif record_type == "medication":
            disease = row.get("disease", "").strip()
            med = row.get("med", "").strip()
            if disease and med:
                disease_meds.setdefault(disease, []).append(
                    {"name": med, "m_days": int(float(row.get("m_days") or 30))}
                )
        elif record_type == "pathway":
            disease = row.get("disease", "").strip()
            stage = int(float(row.get("stage") or 0))
            pathways.setdefault(disease, []).append((stage, split_items(row.get("meds", ""))))
        elif record_type == "retail_product":
            products.append(
                {
                    "product": row.get("product", "").strip(),
                    "curve": row.get("curve", "constant").strip(),
                    "start": float(row.get("start") or 0),
                    "end": float(row.get("end") or 0),
                    "peak": float(row.get("peak") or 0),
                    "sigma": float(row.get("sigma") or 1),
                    "min_p": float(row.get("min_p") or 0),
                    "max_p": float(row.get("max_p") or 0),
                    "base_p": float(row.get("base_p") or 0),
                }
            )
        elif record_type == "retail_modifier":
            disease = row.get("disease", "").strip()
            product = row.get("product", "").strip()
            if disease and product:
                modifiers.setdefault(disease, {})[product] = float(row.get("delta") or 0)

    normalized_pathways = {
        disease: [meds for _, meds in sorted(stages, key=lambda item: item[0])]
        for disease, stages in pathways.items()
    }
    for disease, meds in disease_meds.items():
        if disease not in normalized_pathways:
            normalized_pathways[disease] = [[med["name"]] for med in meds[:1]]

    return {
        "params": params,
        "disease_meds": disease_meds,
        "pathways": normalized_pathways,
        "products": [product for product in products if product["product"]],
        "modifiers": modifiers,
    }


def product_probability(product_config: dict, age: int) -> float:
    curve = product_config["curve"]
    if curve == "decay":
        start, end = product_config["start"], product_config["end"]
        if age <= start:
            return product_config["max_p"]
        if age >= end:
            return product_config["min_p"]
        ratio = (age - start) / (end - start)
        return product_config["max_p"] - ratio * (product_config["max_p"] - product_config["min_p"])
    if curve == "grow":
        start, end = product_config["start"], product_config["end"]
        if age <= start:
            return product_config["min_p"]
        if age >= end:
            return product_config["max_p"]
        ratio = (age - start) / (end - start)
        return product_config["min_p"] + ratio * (product_config["max_p"] - product_config["min_p"])
    if curve == "bell":
        sigma = product_config["sigma"] or 1
        return product_config["max_p"] * math.exp(-0.5 * ((age - product_config["peak"]) / sigma) ** 2)
    return product_config["base_p"]


def draw_products(config: dict, disease: str, age: int, rng: random.Random) -> list[str]:
    probs = {}
    for product_config in config["products"]:
        product = product_config["product"]
        prob = product_probability(product_config, age)
        prob += config["modifiers"].get(disease, {}).get(product, 0.0)
        probs[product] = min(1.0, max(0.0, prob))

    selected = [product for product, prob in probs.items() if rng.random() < prob]
    max_items = int(config["params"].get("max_basket_items", 4))
    if len(selected) > max_items:
        selected = sorted(selected, key=lambda product: probs[product], reverse=True)[:max_items]
    if not selected and probs:
        selected = [max(probs, key=probs.get)]
    return selected


def generate_fake_data(config: dict) -> list[dict]:
    params = config["params"]
    rng = random.Random(int(params.get("seed", 42)))
    n_clients = int(params.get("n_clients", 200))
    simulation_days = int(params.get("simulation_days", 2880))
    diseases = list(config["disease_meds"].keys())
    rows = []

    for client_number in range(1, n_clients + 1):
        client_id = f"C_{client_number:03d}"
        disease = rng.choice(diseases)
        age = int(min(95, max(18, rng.gauss(58, 16))))
        current_day = rng.randint(0, 30)
        stage = 0
        progression_rate = rng.uniform(0.02, 0.15)

        while current_day <= simulation_days:
            stages = config["pathways"][disease]
            if rng.random() < progression_rate and stage < len(stages) - 1:
                stage += 1
            meds = stages[stage]
            med_days = {
                med["name"]: med["m_days"]
                for med in config["disease_meds"].get(disease, [])
            }
            m_days = max([med_days.get(med, 30) for med in meds] or [30])
            pharmacy_delay = min(5, int(rng.expovariate(1 / 2.0)))
            pharmacy_day = current_day + pharmacy_delay
            if pharmacy_day > simulation_days:
                break

            rows.append(
                {
                    "client_id": client_id,
                    "age": age,
                    "disease": disease,
                    "doctor_visit_day": current_day,
                    "pharmacy_report_day": pharmacy_day,
                    "m_days": m_days,
                    "meds_received": "|".join(meds),
                    "retail_products": "|".join(draw_products(config, disease, age, rng)),
                }
            )
            current_day = pharmacy_day + m_days + min(10, int(rng.expovariate(1 / 3.0)))

    return rows


def write_fake_data(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake_data.csv from config_data/config.csv.")
    parser.add_argument("--config", default="config_data/config.csv", help="Input simulator config CSV.")
    parser.add_argument("--output", default="fake_data.csv", help="Generated fake data CSV path.")
    args = parser.parse_args()
    config = read_config(Path(args.config))
    rows = generate_fake_data(config)
    write_fake_data(rows, Path(args.output))
    print(f"Fake data written to {Path(args.output).resolve()} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
