# IMDb Sentiment Classification with Linear SVM

This assignment uses the official Stanford IMDb Large Movie Review Dataset
(25,000 labeled training reviews and 25,000 labeled test reviews) from its
Hugging Face dataset mirror. Dataset originally introduced by Maas et al.
(2011). The script downloads the split files on its first run and keeps the
untouched official test split for evaluation. The dataset is not redistributed
in this repository; it remains subject to its source terms.

## Run

```powershell
py -m pip install -r requirements.txt
py assignment.py
```

Dataset split files are stored in `data/`. Committed results include the
metrics, coefficient table, analysis report, and plot. Running the script also
generates five full error-analysis reviews in
`results/misclassified_reviews.md`; that file is intentionally excluded from
the repository because it reproduces source review text. The report explains
the error-analysis findings, and the local file gives the full examples.

Dataset citation: Maas, A. L. et al. (2011), *Learning Word Vectors for
Sentiment Analysis*, ACL. Dataset mirror:
[stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb).

## Experiment details

- TF-IDF is fit only on the training reviews and uses lowercase word tokens,
  `min_df=2`, `max_df=0.95`, and sublinear term frequency.
- `LinearSVC` is evaluated at `C = 0.01, 0.1, 1, 10`.
- Precision, recall, and F1 use the positive-review class as the positive
  label; accuracy is overall test accuracy.
- The unigram and unigram-plus-bigram experiments use identical splits and
  TF-IDF settings apart from `ngram_range`.
- The ten largest positive coefficients and ten most negative coefficients
  come from the unigram model with the best test F1. Positive coefficients
  favor positive sentiment; negative coefficients favor negative sentiment.
- The error analysis selects five incorrectly classified test reviews that
  contain at least one of those twenty influential unigram features.

Running this assignment repeatedly is deterministic. The test set is used for
model comparison as requested, so the best test C is an evaluation result and
not a substitute for a validation set in production model selection.
