import argparse
import csv
from pathlib import Path


DISEASE_MEDS = {
    "Hypertension": [
        ("Lisinopril", 30),
        ("Amlodipine", 30),
        ("HCTZ", 60),
    ],
    "Type_2_Diabetes": [
        ("Metformin", 60),
        ("Glipizide", 30),
        ("Empagliflozin", 30),
        ("Insulin", 30),
    ],
    "Asthma": [
        ("Inhaled Corticosteroid", 60),
        ("Albuterol Rescue", 90),
        ("Leukotriene Modifier", 30),
    ],
    "Hyperlipidemia": [
        ("Atorvastatin", 90),
        ("Ezetimibe", 30),
        ("PCSK9_Inhibitor", 30),
    ],
    "Rheumatoid_Arthritis": [
        ("Methotrexate", 30),
        ("Folic_Acid", 90),
        ("Adalimumab", 28),
    ],
}

PATHWAYS = {
    "Hypertension": [
        ["Lisinopril"],
        ["Lisinopril", "Amlodipine"],
        ["Lisinopril", "Amlodipine", "HCTZ"],
    ],
    "Type_2_Diabetes": [
        ["Metformin"],
        ["Metformin", "Glipizide"],
        ["Metformin", "Glipizide", "Empagliflozin"],
        ["Metformin", "Empagliflozin", "Insulin"],
    ],
    "Asthma": [
        ["Inhaled Corticosteroid"],
        ["Inhaled Corticosteroid", "Albuterol Rescue", "Leukotriene Modifier"],
    ],
    "Hyperlipidemia": [
        ["Atorvastatin"],
        ["Atorvastatin", "Ezetimibe"],
        ["Atorvastatin", "Ezetimibe", "PCSK9_Inhibitor"],
    ],
    "Rheumatoid_Arthritis": [
        ["Methotrexate", "Folic_Acid"],
        ["Methotrexate", "Folic_Acid", "Adalimumab"],
    ],
}

PRODUCTS = [
    ("Snacks", "decay", 18, 40, "", "", 0.10, 0.80, 0.00),
    ("Condoms", "decay", 18, 45, "", "", 0.00, 0.60, 0.00),
    ("Cosmetics", "decay", 18, 55, "", "", 0.10, 0.50, 0.00),
    ("Coffee", "bell", "", "", 35, 15, 0.00, 0.60, 0.00),
    ("Shampoo", "constant", "", "", "", "", 0.00, 0.00, 0.20),
    ("Supplements", "grow", 25, 70, "", "", 0.10, 0.70, 0.00),
    ("Milk_Powder", "grow", 50, 80, "", "", 0.00, 0.50, 0.00),
    ("Bandages", "grow", 40, 80, "", "", 0.05, 0.20, 0.00),
    ("Adult_Diapers", "grow", 65, 90, "", "", 0.00, 0.60, 0.00),
    ("Sleep_Aids", "bell", "", "", 50, 15, 0.00, 0.30, 0.00),
]

MODIFIERS = [
    ("Hypertension", "Coffee", -0.40),
    ("Hypertension", "Sleep_Aids", 0.30),
    ("Hypertension", "Supplements", 0.20),
    ("Type_2_Diabetes", "Snacks", -0.60),
    ("Type_2_Diabetes", "Bandages", 0.50),
    ("Type_2_Diabetes", "Milk_Powder", 0.30),
    ("Asthma", "Cosmetics", -0.30),
    ("Asthma", "Coffee", 0.10),
    ("Hyperlipidemia", "Supplements", 0.40),
    ("Rheumatoid_Arthritis", "Bandages", 0.40),
    ("Rheumatoid_Arthritis", "Sleep_Aids", 0.30),
    ("Rheumatoid_Arthritis", "Supplements", 0.20),
]

PARAMS = {
    "n_clients": 200,
    "simulation_days": 2880,
    "split_ratio": 0.8,
    "seq_len": 5,
    "seed": 42,
    "max_basket_items": 4,
}


def build_config_rows() -> list[dict]:
    rows = []
    for name, value in PARAMS.items():
        rows.append({"record_type": "parameter", "name": name, "value": value})

    for disease, meds in DISEASE_MEDS.items():
        rows.append({"record_type": "disease", "disease": disease})
        for med_name, m_days in meds:
            rows.append({"record_type": "medication", "disease": disease, "med": med_name, "m_days": m_days})

    for disease, stages in PATHWAYS.items():
        for stage, meds in enumerate(stages):
            rows.append({"record_type": "pathway", "disease": disease, "stage": stage, "meds": "|".join(meds)})

    for product, curve, start, end, peak, sigma, min_p, max_p, base_p in PRODUCTS:
        rows.append(
            {
                "record_type": "retail_product",
                "product": product,
                "curve": curve,
                "start": start,
                "end": end,
                "peak": peak,
                "sigma": sigma,
                "min_p": min_p,
                "max_p": max_p,
                "base_p": base_p,
            }
        )

    for disease, product, delta in MODIFIERS:
        rows.append({"record_type": "retail_modifier", "disease": disease, "product": product, "delta": delta})

    return rows


def write_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_config_rows()
    fieldnames = [
        "record_type",
        "name",
        "value",
        "disease",
        "med",
        "m_days",
        "stage",
        "meds",
        "product",
        "curve",
        "start",
        "end",
        "peak",
        "sigma",
        "min_p",
        "max_p",
        "base_p",
        "delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a simulator config CSV.")
    parser.add_argument("--output", default="config_data/config.csv", help="Config CSV output path.")
    args = parser.parse_args()
    write_config(Path(args.output))
    print(f"Config written to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
