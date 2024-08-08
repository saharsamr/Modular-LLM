import nltk
from nltk.tokenize import word_tokenize
from datasets import load_metric


nltk.download('punkt')


def compute_experts_metrics(labels, predictions):
    bleu_metric = load_metric('bleu', trust_remote_code=True)
    rouge_metric = load_metric('rouge', trust_remote_code=True)
    bert_score = load_metric('bertscore', trust_remote_code=True)

    references = [[word_tokenize(ref)] for ref in labels]
    predictions = [word_tokenize(pred) for pred in predictions]

    bleu_output = bleu_metric.compute(
        predictions=predictions, references=references, max_order=1)['precisions']
    rouge_output = rouge_metric.compute(
        predictions=predictions, references=references, rouge_types=['rougeL'])['rougeL'][1]['fmeasure']
    bertscore_output = bert_score.compute(
        predictions=predictions, references=references, lang='en', model_type='bert-base-uncased')['f1']

    return {
        'bleu': bleu_output,
        'rouge': rouge_output,
        'bertscore': bertscore_output
    }
