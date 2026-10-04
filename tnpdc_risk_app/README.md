# TNPDC Event Risk Predictor

## Hosting on Streamlit Community Cloud
Upload app.py, requirements.txt and all three .joblib files to the root of a GitHub repository. At https://share.streamlit.io choose Create app, select that repository and its branch, and set the entrypoint to app.py. Select Python 3.12 in Advanced settings, then deploy.

## Run on Mac
In Terminal, enter this project directory and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The model records XGBoost 3.2.0 and the label encoder records scikit-learn 1.9.0. These versions are pinned to match the supplied artifacts. If pip cannot find them, obtain the exact training environment or re-export the model with available matching versions; do not silently substitute older packages.

Validation: App startup and a single-meter prediction passed locally using XGBoost 3.2.0 and scikit-learn 1.9.0. Hosted deployment and bulk upload have not been verified.
