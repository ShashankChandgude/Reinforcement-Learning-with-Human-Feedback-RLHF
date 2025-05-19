import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_score


def bleu(predictions, references) -> float:
    """
    Compute corpus BLEU score using sacrebleu.
    """
    # sacrebleu expects references as a list of reference lists
    return sacrebleu.corpus_bleu(predictions, [references]).score


def rouge(predictions, references) -> float:
    """
    Compute average ROUGE-L F1 score across all examples.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(references, predictions)]
    return sum(scores) / len(scores) if scores else 0.0


def bertscore(predictions, references) -> float:
    """
    Compute average BERTScore F1
    """
    P, R, F1 = bert_score_score(predictions, references, lang="en", verbose=False)
    return F1.mean().item()


def compute_all(predictions, references, metrics=None) -> dict[str, float]:
    """
    Compute multiple metrics and return a dict mapping metric names to scores.
    """
    metrics = metrics or ["bleu", "rouge", "bertscore"]
    results = {}
    for m in metrics:
        if m == "bleu":
            results["bleu"] = bleu(predictions, references)
        elif m == "rouge":
            results["rouge"] = rouge(predictions, references)
        elif m == "bertscore":
            results["bertscore"] = bertscore(predictions, references)
        else:
            raise ValueError(f"Unknown metric: {m}")
    return results
