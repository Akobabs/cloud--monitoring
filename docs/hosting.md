# Host the research dashboard

The repository includes a dashboard-only snapshot in `dashboard_data/runs`.
It includes the calibrated baseline and retrained research results, predictions,
metadata and figures. Raw datasets, fitted models and training intermediates are
excluded. Existing local runs take precedence when running on your computer.

## Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with the GitHub account that owns this repository.
2. Create an app using repository `Akobabs/cloud--monitoring` and the branch containing these changes.
3. Set the entry point to `deploy/app.py` (this selects `deploy/requirements.txt`, avoiding training dependencies).
4. Under advanced settings, select Python 3.12, then deploy.

Hosting displays saved experiments. It does not run training or connect to live cloud telemetry.
The prediction chart uses a direct Vega-Lite specification to avoid the Altair
5.5 import incompatibility with Python 3.14. Python 3.12 remains the recommended
runtime to match the research environment.
The bundled Google-derived results retain the source's noncommercial attribution/share-alike caveat;
see the README for dataset provenance and interpretation.

## Other Python hosts

Install `requirements-dashboard.txt` with Python 3.12, then start:

```sh
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Set the port to the value required by your provider. Use the provider's HTTPS endpoint.
The dashboard requires no secrets or database.

## Refresh the published results

After generating and reviewing the research results locally, run:

```sh
python scripts/export_dashboard.py
```

Commit the updated `dashboard_data` files and redeploy. The original training
requirements and Windows research launchers are unchanged.
