from scipy.stats import pearsonr


def pearson_metrics(tokenizer, eval_preds):
    predictions, labels = eval_pred

    if predictions.ndim == 2:
        predictions = predictions[:, 0]  # 필요에 따라 차원 축소

    # 피어슨 상관계수 계산
    pearson_corr, _ = pearsonr(predictions, labels)

    return {
        'pearson': pearson_corr,
    }
