# Housing Affordability Clustering

This project generates a synthetic Bengaluru housing dataset, performs unsupervised clustering to find affordability groups, and serves an interactive Streamlit app.

## Quickstart

1. Create virtual env & install
```
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Generate data
```
python data/generate_data.py --n 2000 --out data/housing_bengaluru_realistic.csv
```

3. Train (produces outputs/ and models/kmeans_model.joblib)
```
python notebooks/train_clustering_pipeline.py --input data/housing_bengaluru_realistic.csv --out outputs --use db
```

4. Run Streamlit app
```
streamlit run app/streamlit_app.py
```

## Notes
- `train_clustering_pipeline.py` lives in `notebooks/` but you can move it to `scripts/` if you prefer.
- For production, replace synthetic data with municipal datasets and consider geospatial clustering techniques.