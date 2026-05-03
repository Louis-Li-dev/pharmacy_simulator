# Pharmacy Simulator

A Flask-based clinical pathway simulation and prediction dashboard for patient medication refill behavior. The app generates synthetic patient timelines, trains a baseline LSTM model, and provides an interactive browser UI with plots and patient-level prediction summaries.

## Features

- Generate synthetic clinical pathway data for multiple patients
- Visualize patient timelines and medication coverage
- Train a PyTorch LSTM model on generated sequence data
- Compare predicted next refill timing against actual patient events
- Interactive UI with Tailwind CSS and ECharts charts
- API endpoints for data generation, model training, and patient-level JSON output

## Repository Structure

- `app.py` - Flask application and REST endpoints
- `module/data_generator.py` - Synthetic clinical pathway generation and preprocessing
- `module/model.py` - PyTorch model, statistical baseline, and prediction utilities
- `module/data.py` - Dataset wrapper for training and evaluation
- `module/plot.py` - Plot utilities for model comparison
- `templates/index.html` - Frontend dashboard with Tailwind and ECharts

## Requirements

- Python 3.11+ recommended
- `Flask`
- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## Setup

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

```bash
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python app.py
```

5. Open a browser and navigate to:

```
http://127.0.0.1:5000
```

## Usage

- Use the dashboard to generate synthetic patient data.
- Train the model with the provided parameters.
- Once training completes, explore patient-level predictions using the second tab.

## Vercel Deployment

This repository includes `vercel.json` for Vercel deployment. It uses the Python runtime to serve `app.py` as the application entrypoint.

To deploy with Vercel:

```bash
vercel login
vercel
```

## Notes

- The app generates data in memory and does not persist datasets between restarts.
- Model training is performed in-process and can take longer depending on available compute and dataset size.
- The notebook files are included for experimentation and data generation, but the Flask app is the application entrypoint.
