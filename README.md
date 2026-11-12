# Airline Customer Experience Intelligence

Analyses airline customer tweets: sentiment (TF-IDF + Logistic Regression), calibrated probabilities, chi-square and correlation tests on issue categories, and an exact optimiser that picks which complaints to handle within a support-time budget. Results are shown in a simple static dashboard (plain HTML + Plotly.js) that can be hosted on Vercel.

## 1. Dataset to download

**Use: Twitter US Airline Sentiment** (Kaggle, `crowdflower/twitter-airline-sentiment`)

- File you need: `Tweets.csv` (about 14,640 tweets, 6 US airlines, February 2015)
- Labels: `airline_sentiment` = positive / neutral / negative (matches your "positive, neutral, negative" requirement)
- Issue categories: `negativereason` (e.g. Late Flight, Lost Luggage, Customer Service Issue)
- Licence: CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution)

Download with the Kaggle CLI (needs a free Kaggle account and API token in `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d crowdflower/twitter-airline-sentiment -p data --unzip
```

Or download it manually from the Kaggle dataset page and place `Tweets.csv` in the `data/` folder.

The screenshot's dataset (British Airways Skytrax reviews) is an alternative, but its exact columns and ground-service field could not be confirmed, so this project uses the Twitter dataset.

## 2. Setup and run

```bash
pip install -r requirements.txt
python -m src.pipeline                 # reads data/Tweets.csv, writes public/data/results.json
```

Optional: open `public/index.html` in a browser after the pipeline runs. Browsers block `fetch` on `file://`, so use `python -m http.server -d public 8000` and open http://localhost:8000.

## 3. What each part does

| Component | Module | Method |
|---|---|---|
| Cleaning | `src/preprocess.py` | Drops duplicates and rows missing text/sentiment, removes URLs/mentions/emojis |
| Sentiment + probabilities | `src/sentiment.py` | TF-IDF (1-2 grams) + Logistic Regression, sigmoid calibration, 5-fold out-of-fold predictions; precision, recall, F1, confusion matrix, Brier score, calibration curve |
| Statistics | `src/stats.py` | Chi-square test per issue vs negative sentiment, Cramér's V, point-biserial correlation, chi-square of negative reasons across airlines |
| Optimisation | `src/prioritize.py` | 0/1 knapsack: maximise severity subject to support minutes, solved exactly with `scipy.optimize.milp` |
| Dashboard | `public/index.html` | Airline filter, capacity slider, Plotly charts |

## 4. Assumptions to review

These are not in the dataset. Edit them in `src/config.py`:

- `ISSUE_WEIGHT`: business impact of each issue category
- `ISSUE_MINUTES`: average agent minutes per complaint
- `CAPACITY_STEPS`: support-time levels shown on the slider
- `ISSUE_KEYWORDS`: keyword rules that assign issue categories to tweet text

## 5. Deploy to Vercel

Vercel hosts the static dashboard in `public/`. Vercel serverless functions are not suited to running Streamlit, so the Streamlit option from the original brief is replaced by this static dashboard.

1. Run `python -m src.pipeline` locally so `public/data/results.json` exists.
2. Push the project to GitHub (`data/*.csv` is git-ignored, so the raw data is not uploaded).
   The results file is committed, so the dashboard works without the raw data.
3. In Vercel: **Add New → Project**, import the GitHub repo.
   Framework preset: *Other*. Build command: empty. Output directory: `public` (already set in `vercel.json`).
4. Deploy. Future changes: rerun the pipeline, commit `public/data/results.json`, and push.

Or from the terminal: `npm i -g vercel` then `vercel` (preview) and `vercel --prod` (production).

## 6. Project layout

```
airline/
├── data/                  # put Tweets.csv here (git-ignored)
├── public/
│   ├── index.html         # dashboard
│   └── data/results.json  # written by the pipeline
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── sentiment.py
│   ├── stats.py
│   ├── prioritize.py
│   └── pipeline.py
├── requirements.txt
└── vercel.json
```
