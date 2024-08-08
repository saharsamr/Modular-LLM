from datasets import load_metric
import numpy as np


def compute_experts_metrics(labels, predictions):
    print(labels)
    print(predictions)
    bleu_metric = load_metric('bleu', trust_remote_code=True)
    rouge_metric = load_metric('rouge', trust_remote_code=True)
    bert_score = load_metric('bertscore', trust_remote_code=True)

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
