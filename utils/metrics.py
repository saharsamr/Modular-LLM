import nltk
from nltk.tokenize import word_tokenize
import evaluate
import numpy as np
import logging


nltk.download('punkt')

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)


def compute_experts_metrics(labels, predictions):
    bleu_metric = evaluate.load('bleu', trust_remote_code=True)
    rouge_metric = evaluate.load('rouge', trust_remote_code=True)
    bert_score = evaluate.load('bertscore', trust_remote_code=True)

    references = [[word_tokenize(ref)] for ref in labels]
    predictions = [word_tokenize(pred) for pred in predictions]

    bleu_output = bleu_metric.compute(
        predictions=predictions, references=references, max_order=1)['precisions'][0]
    rouge_output = rouge_metric.compute(
        predictions=predictions, references=references, rouge_types=['rougeL'])['rougeL'][1][2]
    bertscore_output = bert_score.compute(
        predictions=predictions, references=references, lang='en', model_type='bert-base-uncased')['f1']
    bertscore_output = np.mean(bertscore_output)

    return {
        'bleu': bleu_output,
        'rouge': rouge_output,
        'bertscore': bertscore_output
    }
