import pytest
import torch
import sacrebleu
from rouge_score import rouge_scorer
from evaluation import eval_metrics as metrics

class DummyScore:
    def __init__(self, score):
        self.score = score

@pytest.fixture(autouse=True)
def patch_metric_backends(monkeypatch):
    # Patch sacrebleu
    monkeypatch.setattr(
        sacrebleu, 'corpus_bleu',
        lambda preds, refs: DummyScore(42)
    )
    # Patch RougeScorer
    class FakeRougeScorer:
        def __init__(self, names, use_stemmer): pass
        def score(self, ref, pred):
            return {'rougeL': type('X', (), {'fmeasure': 0.5})()}
    monkeypatch.setattr(
        rouge_scorer, 'RougeScorer',
        FakeRougeScorer
    )
    # Patch bert-score
    monkeypatch.setattr(
        metrics, 'bert_score_score',
        lambda preds, refs, lang, verbose: (None, None, torch.tensor([0.75]))
    )
    yield

def test_bleu():
    score = metrics.bleu(['foo'], ['bar'])
    assert score == 42

def test_rouge():
    score = metrics.rouge(['foo'], ['bar'])
    assert pytest.approx(score, rel=1e-3) == 0.5

def test_bertscore():
    score = metrics.bertscore(['foo'], ['bar'])
    assert pytest.approx(score, rel=1e-3) == 0.75

def test_compute_all():
    result = metrics.compute_all(['a'], ['b'], metrics=['bleu','rouge','bertscore'])
    assert result['bleu'] == 42
    assert pytest.approx(result['rouge'], rel=1e-3) == 0.5
    assert pytest.approx(result['bertscore'], rel=1e-3) == 0.75
    with pytest.raises(ValueError):
        metrics.compute_all(['x'], ['y'], metrics=['unknown'])
