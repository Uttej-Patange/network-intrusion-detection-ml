# Network Intrusion Detection, Multi-Class Classification

## a. Problem Statement

Network traffic needs to be classified as normal or as one of several attack categories so that intrusions can be flagged automatically instead of relying on manual inspection. This project trains and compares five classification models that take a set of connection-level features (duration, byte counts, protocol, service, error rates, and related traffic statistics) and predict the traffic type. The trained models are served through an interactive Streamlit application where a user can upload test data, choose a model, and review its predictions and evaluation metrics.

## b. Dataset Description

The dataset is the **KDD Cup 1999 network intrusion detection dataset** (10 percent subset), sourced via `sklearn.datasets.fetch_kddcup99`. The original subset contains 494,021 connection records with 41 features (well above the assignment's minimum of 500 instances and 12 features).

Each record is labeled with one of 23 fine-grained attack types, which were grouped into 5 broad categories for this project, following the standard KDD Cup taxonomy:

| Category | Meaning | Original labels included |
|---|---|---|
| normal | Legitimate traffic | normal |
| dos | Denial of Service | back, land, neptune, pod, smurf, teardrop |
| probe | Surveillance / scanning | ipsweep, nmap, portsweep, satan |
| r2l | Remote to Local, unauthorized access from a remote machine | ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster |
| u2r | User to Root, unauthorized access to root privileges | buffer_overflow, loadmodule, perl, rootkit |

The raw dataset is extremely imbalanced (`dos` alone accounts for roughly 79 percent of records, while `u2r` accounts for barely 0.01 percent). To keep the trained model files small enough to host on GitHub and deploy on the Streamlit Community Cloud free tier, the two dominant classes (`dos` and `normal`) were capped at 50,000 records each through random sampling, while every record from the rare `probe`, `r2l`, and `u2r` classes was kept in full. This produced a working dataset of **105,285 rows**, still well above the required minimum and still meaningfully imbalanced, which is realistic for an intrusion detection setting and is discussed further in the observations below.

Features used:
- **Categorical (3):** `protocol_type`, `service`, `flag`
- **Numeric (38):** connection duration, byte counts, error and rerror rates, host and service counts, and related traffic statistics
- **Target:** `attack_category`, one of `normal`, `dos`, `probe`, `r2l`, `u2r`

The 20 percent held-out test split (used for the metrics below) and a further 3,000-row stratified sample (`test_data.csv`) used for the Streamlit app's upload feature both preserve the original class proportions.

## c. GitHub Repository Link

https://github.com/Uttej-Patange/network-intrusion-detection-ml

## Live Streamlit App Link

https://network-intrusion-detection-ml-67lq3z8qnv6mkisjv3knkd.streamlit.app/

## d. Models Used

All five models were trained on the same preprocessed dataset (median imputation and standard scaling for numeric features, one-hot encoding for categorical features) with an 80/20 stratified train-test split. Because the target has 5 classes, Precision, Recall, and F1 are macro-averaged (unweighted mean across all 5 classes) and AUC is computed one-vs-rest, also macro-averaged, so that rare classes are not drowned out by the dominant ones.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9974 | 0.9942 | 0.9440 | 0.9220 | 0.9325 | 0.9952 |
| Decision Tree | 0.9989 | 0.9672 | 0.9510 | 0.9343 | 0.9422 | 0.9979 |
| kNN | 0.9976 | 0.9683 | 0.8839 | 0.8849 | 0.8841 | 0.9957 |
| Naive Bayes | 0.8222 | 0.9394 | 0.5283 | 0.7524 | 0.5234 | 0.7374 |
| Random Forest | 0.9985 | 0.9999 | 0.9651 | 0.8840 | 0.9160 | 0.9972 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong all-round performance despite being the simplest model, which suggests the classes are close to linearly separable once the features are scaled and encoded. The gap between Accuracy (0.9974) and macro Recall (0.9220) shows it still misses a meaningful share of the rare attack types. |
| Decision Tree | The best balance of Precision, Recall, and F1 among all five models, and the highest MCC (0.9979). A single tree can carve out the tight, rule-like boundaries that separate rare attack types such as `r2l` and `u2r` from `normal` traffic, which suits this dataset well. |
| kNN | High accuracy but the lowest Precision, Recall, and F1 among the non-Naive-Bayes models. With `u2r` having only a handful of training examples, its neighborhoods are easily swamped by nearby majority-class points, so it struggles specifically on the rarest classes even though overall accuracy looks fine. |
| Naive Bayes | Clearly the weakest model. Its accuracy (0.8222) and macro F1 (0.5234) are far below the others because the Gaussian independence assumption does not hold well for correlated traffic features like the various rate and count columns, causing it to over-predict certain classes. |
| Random Forest | Highest AUC (0.9999) and second-highest MCC (0.9972), confirming it ranks and separates classes very well even though its macro Recall (0.8840) trails the Decision Tree slightly, most likely on the rarest `u2r` class where very few training examples are available to any individual tree in the ensemble. |

**Overall winner:** **Decision Tree**, based on the highest MCC (0.9979) combined with the best macro Precision, Recall, and F1 across all 5 classes. MCC and macro-averaged metrics were prioritized over raw Accuracy or AUC because this dataset is heavily imbalanced (`u2r` has only 52 records in the full dataset), and Accuracy alone can look excellent even when a model fails completely on the rare, arguably most important, attack categories.

## Project Structure

```
project-folder/
├── app.py                        # Streamlit application
├── requirements.txt
├── README.md
├── test_data.csv                 # 3,000-row stratified sample for the app's upload feature
├── data/
│   └── kddcup_processed.csv      # Capped working dataset (105,285 rows)
└── model/
    ├── train.py                  # Trains all 5 models and writes the files below
    ├── *.joblib                  # Saved preprocessing + model pipelines (one per model)
    ├── metrics_comparison.csv    # The comparison table above, generated from the held-out test split
    └── feature_schema.json       # Feature lists and class labels used by app.py
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train.py       # retrains all 5 models from scratch (downloads the dataset)
streamlit run app.py
```
