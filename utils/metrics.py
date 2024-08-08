from datasets import load_metric
import numpy as np


def compute_experts_metrics(eval_preds):
    bleu_metric = load_metric('bleu')
    rouge_metric = load_metric('rouge')
    bert_score = load_metric('bertscore')

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    bleu_output = bleu_metric.compute(
        predictions=predictions, references=labels, max_order=1)
    rouge_output = rouge_metric.compute(
        predictions=predictions, references=labels, rouge_types=['rougeL'])
    bertscore_output = bert_score.compute(
        predictions=predictions, references=labels, lang='en', model_type='bert-base-uncased')

    return {
        'bleu': bleu_output,
        'rouge': rouge_output,
        'bertscore': bertscore_output
    }
