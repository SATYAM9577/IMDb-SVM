# IMDb Sentiment Classification with Linear SVM

## Data and method

Used the official Stanford IMDb Large Movie Review Dataset: 25,000 labeled training reviews and 25,000 labeled test reviews. The predefined training/test split was preserved. A training-only TF-IDF vocabulary was built with lowercase Unicode-normalized tokens, `min_df=2`, `max_df=0.95`, and sublinear term frequency. LinearSVC was evaluated at C values 0.01, 0.1, 1, and 10. Precision, recall, and F1 treat positive sentiment as the positive class.

## (c) Test metrics for all C values

| Representation | C | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| unigram | 0.01 | 0.8616 | 0.8476 | 0.8818 | 0.8643 |
| unigram | 0.1 | 0.8892 | 0.8880 | 0.8908 | 0.8894 |
| unigram | 1 | 0.8784 | 0.8854 | 0.8694 | 0.8773 |
| unigram | 10 | 0.8560 | 0.8652 | 0.8434 | 0.8542 |
| unigram+bigrams | 0.01 | 0.8614 | 0.8511 | 0.8759 | 0.8633 |
| unigram+bigrams | 0.1 | 0.8980 | 0.8927 | 0.9048 | 0.8987 |
| unigram+bigrams | 1 | 0.9066 | 0.9082 | 0.9046 | 0.9064 |
| unigram+bigrams | 10 | 0.9032 | 0.9061 | 0.8996 | 0.9028 |

## (d) Best C and F1 plot

The best unigram test F1 is **0.8894** at **C=0.1**. Across both representations, the best result is unigram+bigrams at **C=1** (F1=0.9064). The plot compares both representations. See [`f1_vs_c.png`](f1_vs_c.png).

## (e) Unigrams versus unigrams+bigrams

- Best unigram model: C=0.1, accuracy=0.8892, precision=0.8880, recall=0.8908, F1=0.8894.
- Best unigram+bigrams model: C=1, accuracy=0.9066, precision=0.9082, recall=0.9046, F1=0.9064.

Bigrams can represent short contextual phrases (for example, negated expressions) that a unigram representation cannot distinguish. They also increase dimensionality and sparsity. At their best C values, unigram+bigrams raises F1 by +0.0170 and accuracy by +0.0174 relative to unigrams on this test set.

## (f)-(g) Most influential unigram features

Coefficients come from the best unigram model. Positive coefficients push predictions toward positive sentiment; negative coefficients push toward negative sentiment. Exact signed values are in [`feature_coefficients.csv`](feature_coefficients.csv).

| Positive rank | Feature | Coefficient | Negative rank | Feature | Coefficient |
|---:|---|---:|---:|---|---:|
| 1 | `great` | +2.658008 | 1 | `worst` | -3.246577 |
| 2 | `excellent` | +2.262493 | 2 | `bad` | -2.765782 |
| 3 | `best` | +1.800667 | 3 | `awful` | -2.287253 |
| 4 | `perfect` | +1.748896 | 4 | `waste` | -2.197962 |
| 5 | `wonderful` | +1.610788 | 5 | `boring` | -2.065585 |
| 6 | `amazing` | +1.481792 | 6 | `poor` | -1.966385 |
| 7 | `well` | +1.435750 | 7 | `nothing` | -1.763897 |
| 8 | `fun` | +1.407912 | 8 | `terrible` | -1.664510 |
| 9 | `today` | +1.374231 | 9 | `worse` | -1.604853 |
| 10 | `loved` | +1.362018 | 10 | `poorly` | -1.571314 |

## (h) Intuitive interpretation

Inspect the highest positive and lowest negative coefficients above. Words strongly associated with praise, enjoyment, or quality are expected on the positive side; words associated with dislike, poor quality, or disappointment are expected on the negative side. The top positive features `great`, `excellent`, and `wonderful`, and negative features `worst`, `awful`, and `boring`, are intuitive. `today` and `well` are less sentiment-specific; their appearance likely reflects correlations or contexts in this review corpus. Ambiguous coefficients describe the fitted corpus, not universal word sentiment.

## (i) Error analysis

Found **2135** eligible misclassified test reviews containing at least one of the strongest positive or negative features. Five full reviews with matched feature contributions and case-specific explanations are generated locally in `results/misclassified_reviews.md` (excluded from version control because it reproduces dataset review text). The linear TF-IDF model ignores much long-distance syntax and world knowledge; negation, sarcasm, mixed opinions, plot details, and a few high-weight terms can make the summed feature evidence conflict with the review's overall label.
