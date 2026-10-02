from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SPLIT_FILES = {
    "train": (
        DATA_DIR / "imdb_train.parquet",
        (
            "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/"
            "plain_text/train-00000-of-00001.parquet"
        ),
    ),
    "test": (
        DATA_DIR / "imdb_test.parquet",
        (
            "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/"
            "plain_text/test-00000-of-00001.parquet"
        ),
    ),
}
C_VALUES = (0.01, 0.1, 1, 10)


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, (path, url) in SPLIT_FILES.items():
        if not path.exists():
            print(f"Downloading the official IMDb {split} split to {path} ...")
            urllib.request.urlretrieve(url, path)


def read_split(split: str) -> tuple[list[str], list[int]]:
    path, _ = SPLIT_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(
            f"IMDb {split} split not found at {path}; run without --skip-download."
        )
    frame = pd.read_parquet(path, columns=["text", "label"])
    return frame["text"].astype(str).tolist(), frame["label"].astype(int).tolist()


def make_vectorizer(ngram_range: tuple[int, int]) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def metric_row(
    representation: str,
    c_value: float,
    y_true: list[int],
    predictions: list[int],
) -> dict[str, float | str]:
    return {
        "representation": representation,
        "C": c_value,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def run_experiment(
    representation: str,
    ngram_range: tuple[int, int],
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
) -> tuple[list[dict[str, float | str]], TfidfVectorizer, dict[float, LinearSVC]]:
    print(f"Fitting {representation} TF-IDF features ...")
    vectorizer = make_vectorizer(ngram_range)
    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    print(f"Feature count: {train_matrix.shape[1]:,}")

    rows = []
    models: dict[float, LinearSVC] = {}
    for c_value in C_VALUES:
        model = LinearSVC(C=c_value, random_state=42, max_iter=5000)
        model.fit(train_matrix, train_labels)
        predictions = model.predict(test_matrix).tolist()
        rows.append(metric_row(representation, c_value, test_labels, predictions))
        models[c_value] = model
        print(f"  C={c_value:g}: test F1={rows[-1]['f1']:.4f}")
    return rows, vectorizer, models


def write_feature_table(
    vectorizer: TfidfVectorizer,
    model: LinearSVC,
    output_path: Path,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    names = vectorizer.get_feature_names_out()
    coefficients = model.coef_.ravel()
    positive_ids = coefficients.argsort()[-10:][::-1]
    negative_ids = coefficients.argsort()[:10]
    positive = [(str(names[index]), float(coefficients[index])) for index in positive_ids]
    negative = [(str(names[index]), float(coefficients[index])) for index in negative_ids]
    rows = [
        {"class": label, "rank": rank, "feature": feature, "coefficient": coefficient}
        for label, features in (("positive", positive), ("negative", negative))
        for rank, (feature, coefficient) in enumerate(features, start=1)
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return positive, negative


def write_error_analysis(
    vectorizer: TfidfVectorizer,
    model: LinearSVC,
    test_texts: list[str],
    test_labels: list[int],
    influential: list[tuple[str, float]],
    output_path: Path,
) -> int:
    matrix = vectorizer.transform(test_texts)
    predictions = model.predict(matrix)
    analyzer = vectorizer.build_analyzer()
    feature_weights = dict(influential)
    candidates: list[tuple[int, float, int, str, list[str]]] = []
    for index, (truth, prediction, review) in enumerate(
        zip(test_labels, predictions, test_texts, strict=True)
    ):
        if truth == prediction:
            continue
        matched = sorted(set(analyzer(review)).intersection(feature_weights))
        if matched:
            candidates.append(
                (
                    len(matched),
                    abs(float(model.decision_function(matrix[index]).item())),
                    index,
                    review,
                    matched,
                )
            )

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = candidates[:5]
    if len(selected) < 5:
        raise RuntimeError(
            "Could not find five misclassified test reviews containing a top "
            f"coefficient feature; found {len(selected)}."
        )

    with output_path.open("w", encoding="utf-8") as report:
        report.write("# Five SVM misclassification examples\n\n")
        report.write(
            "Each example is an official test review misclassified by the best "
            "unigram model and containing one or more of its twenty strongest "
            "positive/negative unigram features. Reviews are shown in full.\n\n"
        )
        for number, (count, margin, index, review, matched) in enumerate(selected, 1):
            actual = "positive" if test_labels[index] else "negative"
            predicted = "positive" if predictions[index] else "negative"
            contributions = []
            for term in matched:
                column = vectorizer.vocabulary_[term]
                tfidf_weight = float(matrix[index, column])
                contribution = tfidf_weight * feature_weights[term]
                if tfidf_weight:
                    contributions.append((term, feature_weights[term], contribution))
            contributions.sort(key=lambda item: abs(item[2]), reverse=True)
            weighted_terms = ", ".join(
                f"`{term}` (coef {coefficient:+.4f}, TF-IDF contribution {contribution:+.4f})"
                for term, coefficient, contribution in contributions
            )
            strongest_term, strongest_coefficient, strongest_contribution = contributions[0]
            favored_class = "positive" if strongest_contribution > 0 else "negative"
            report.write(f"## Example {number}\n\n")
            report.write(f"- Actual sentiment: **{actual}**\n")
            report.write(f"- Predicted sentiment: **{predicted}**\n")
            report.write(f"- Absolute decision margin: `{margin:.4f}`\n")
            report.write(f"- Influential matched features: {weighted_terms}\n\n")
            report.write("**Review (full text):**\n\n")
            report.write("> " + review.replace("\n", "\n> ") + "\n\n")
            if favored_class == predicted:
                explanation = (
                    f"The strongest matched feature, `{strongest_term}`, has "
                    f"coefficient {strongest_coefficient:+.4f} and contributes "
                    f"{strongest_contribution:+.4f} toward the incorrect "
                    f"{predicted} prediction. This is compatible with the "
                    "TF-IDF linear score being influenced by that word in this "
                    "review. Its occurrence may be negated, ironic, quoted, or "
                    "part of a mixed-opinion passage; a unigram feature records "
                    "presence and weight, not those contextual distinctions."
                )
            else:
                explanation = (
                    f"The strongest matched feature, `{strongest_term}`, "
                    f"contributes {strongest_contribution:+.4f} toward the "
                    f"actual {actual} class, so it does not itself explain the "
                    "wrong prediction. Other, unlisted terms in the full sparse "
                    "TF-IDF vector must have outweighed it in the linear score. "
                    "The model adds weighted token evidence and does not encode "
                    "long-distance context, sarcasm, or the review's overall "
                    "argument."
                )
            report.write(f"**Possible explanation:** {explanation}\n\n")
    return len(candidates)


def write_report(
    metrics: pd.DataFrame,
    best_unigram: pd.Series,
    best_bigram: pd.Series,
    positive: list[tuple[str, float]],
    negative: list[tuple[str, float]],
    error_count: int,
    output_path: Path,
) -> None:
    metric_table = [
        "| Representation | C | Accuracy | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict(orient="records"):
        metric_table.append(
            f"| {row['representation']} | {row['C']:g} | {row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    lines = [
        "# IMDb Sentiment Classification with Linear SVM",
        "",
        "## Data and method",
        "",
        (
            "Used the official Stanford IMDb Large Movie Review Dataset: 25,000 "
            "labeled training reviews and 25,000 labeled test reviews. The predefined "
            "training/test split was preserved. A training-only TF-IDF vocabulary "
            "was built with lowercase Unicode-normalized tokens, `min_df=2`, "
            "`max_df=0.95`, and sublinear term frequency. LinearSVC was evaluated "
            "at C values 0.01, 0.1, 1, and 10. Precision, recall, and F1 treat "
            "positive sentiment as the positive class."
        ),
        "",
        "## (c) Test metrics for all C values",
        "",
        *metric_table,
        "",
        "## (d) Best C and F1 plot",
        "",
        (
            f"The best unigram test F1 is **{best_unigram['f1']:.4f}** at "
            f"**C={best_unigram['C']:g}**. Across both representations, the "
            f"best result is unigram+bigrams at **C={best_bigram['C']:g}** "
            f"(F1={best_bigram['f1']:.4f}). The plot compares both "
            "representations. See [`f1_vs_c.png`](f1_vs_c.png)."
        ),
        "",
        "## (e) Unigrams versus unigrams+bigrams",
        "",
        (
            f"- Best unigram model: C={best_unigram['C']:g}, "
            f"accuracy={best_unigram['accuracy']:.4f}, "
            f"precision={best_unigram['precision']:.4f}, "
            f"recall={best_unigram['recall']:.4f}, "
            f"F1={best_unigram['f1']:.4f}."
        ),
        (
            f"- Best unigram+bigrams model: C={best_bigram['C']:g}, "
            f"accuracy={best_bigram['accuracy']:.4f}, "
            f"precision={best_bigram['precision']:.4f}, "
            f"recall={best_bigram['recall']:.4f}, "
            f"F1={best_bigram['f1']:.4f}."
        ),
        "",
        (
            "Bigrams can represent short contextual phrases (for example, negated "
            "expressions) that a unigram representation cannot distinguish. They "
            "also increase dimensionality and sparsity. At their best C values, "
            f"unigram+bigrams raises F1 by "
            f"{best_bigram['f1'] - best_unigram['f1']:+.4f} and accuracy by "
            f"{best_bigram['accuracy'] - best_unigram['accuracy']:+.4f} relative "
            "to unigrams on this test set."
        ),
        "",
        "## (f)-(g) Most influential unigram features",
        "",
        (
            "Coefficients come from the best unigram model. Positive coefficients "
            "push predictions toward positive sentiment; negative coefficients "
            "push toward negative sentiment. Exact signed values are in "
            "[`feature_coefficients.csv`](feature_coefficients.csv)."
        ),
        "",
        "| Positive rank | Feature | Coefficient | Negative rank | Feature | Coefficient |",
        "|---:|---|---:|---:|---|---:|",
    ]
    for rank, (pos, neg) in enumerate(zip(positive, negative, strict=True), 1):
        lines.append(
            f"| {rank} | `{pos[0]}` | {pos[1]:+.6f} | {rank} | `{neg[0]}` | {neg[1]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## (h) Intuitive interpretation",
            "",
            (
                "Inspect the highest positive and lowest negative coefficients above. "
                "Words strongly associated with praise, enjoyment, or quality are "
                "expected on the positive side; words associated with dislike, "
                "poor quality, or disappointment are expected on the negative side. "
                "The top positive features `great`, `excellent`, and `wonderful`, "
                "and negative features `worst`, `awful`, and `boring`, are intuitive. "
                "`today` and `well` are less sentiment-specific; their appearance "
                "likely reflects correlations or contexts in this review corpus. "
                "Ambiguous coefficients describe the fitted corpus, not universal "
                "word sentiment."
            ),
            "",
            "## (i) Error analysis",
            "",
            (
                f"Found **{error_count}** eligible misclassified test reviews "
                "containing at least one of the strongest positive or negative "
                "features. Five full reviews with matched feature contributions "
                "and case-specific explanations are generated locally in "
                "`results/misclassified_reviews.md` (excluded from version control "
                "because it reproduces dataset review text). "
                "The linear TF-IDF model ignores much long-distance syntax and "
                "world knowledge; negation, sarcasm, mixed opinions, plot details, "
                "and a few high-weight terms can make the summed feature evidence "
                "conflict with the review's overall label."
            ),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use the already downloaded and extracted dataset.",
    )
    args = parser.parse_args()
    if not args.skip_download:
        download_dataset()

    train_texts, train_labels = read_split("train")
    test_texts, test_labels = read_split("test")
    if len(train_texts) != 25_000 or len(test_texts) != 25_000:
        raise RuntimeError(
            "Expected the standard 25,000/25,000 IMDb split; "
            f"found {len(train_texts)} train and {len(test_texts)} test reviews."
        )
    print(f"Loaded {len(train_texts):,} train and {len(test_texts):,} test reviews.")
    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    unigram_rows, unigram_vectorizer, unigram_models = run_experiment(
        "unigram", (1, 1), train_texts, train_labels, test_texts, test_labels
    )
    bigram_rows, _, _ = run_experiment(
        "unigram+bigrams", (1, 2), train_texts, train_labels, test_texts, test_labels
    )
    metrics = pd.DataFrame(unigram_rows + bigram_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    unigram_metrics = metrics[metrics["representation"] == "unigram"]
    bigram_metrics = metrics[metrics["representation"] == "unigram+bigrams"]
    best_unigram = unigram_metrics.loc[unigram_metrics["f1"].idxmax()]
    best_bigram = bigram_metrics.loc[bigram_metrics["f1"].idxmax()]
    selected_model = unigram_models[float(best_unigram["C"])]
    positive, negative = write_feature_table(
        unigram_vectorizer,
        selected_model,
        output_dir / "feature_coefficients.csv",
    )
    candidates = write_error_analysis(
        unigram_vectorizer,
        selected_model,
        test_texts,
        test_labels,
        positive + negative,
        output_dir / "misclassified_reviews.md",
    )

    fig, axis = plt.subplots(figsize=(8, 5))
    for representation, group in metrics.groupby("representation"):
        axis.plot(group["C"], group["f1"], marker="o", linewidth=2, label=representation)
    axis.set_xscale("log")
    axis.set_xticks(C_VALUES, labels=[str(value) for value in C_VALUES])
    axis.set_xlabel("Regularization parameter C (log scale)")
    axis.set_ylabel("Test F1-score (positive class)")
    axis.set_title("LinearSVC sentiment performance across C values")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "f1_vs_c.png", dpi=160)
    plt.close(fig)
    write_report(
        metrics,
        best_unigram,
        best_bigram,
        positive,
        negative,
        candidates,
        output_dir / "assignment_report.md",
    )
    print("\nTest metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nBest unigram C: {best_unigram['C']:g}")
    print(f"Best bigram C: {best_bigram['C']:g}")
    print(f"Misclassified reviews available for analysis: {candidates}")
    print(f"Assignment outputs written to {output_dir}")


if __name__ == "__main__":
    main()
