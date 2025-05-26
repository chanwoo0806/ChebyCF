import time
import logging
from tqdm import tqdm
import torch
import numpy as np
from src.metric import recall, ndcg, precision, mrr

def train(model, train_data, device):
    start_time = time.time()
    model.fit(train_data) # No convergence needed for training-free models
    model.to(device)
    logging.info('[TRAIN]')
    logging.info(f"Completed in {time.time() - start_time:.2f} seconds.")
    logging.info('')

def test(model, test_dataloader, metrics, top_ks):
    model.eval()
    result = np.zeros((len(metrics), len(top_ks)), dtype=np.float32)
    for observed_inter, label_inter in tqdm(test_dataloader, desc='Testing', total=len(test_dataloader)):
        # Predict preference scores for all items and mask observed interactions
        with torch.no_grad():
            pred_score = model.full_predict(observed_inter)
        # Rank top-k items based on the scores
        _, ranked_items = torch.topk(pred_score, k=max(top_ks))
        # Check if the ranked items are in the label interactions
        relevance = [[item in label for item in ranked.tolist()]
                     for ranked, label in zip(ranked_items, label_inter)]
        relevance = np.array(relevance, dtype=np.float32)
        # Calculate metrics
        for i, metric in enumerate(metrics):
            for j, k in enumerate(top_ks):
                if metric == 'recall':
                    result[i, j] += recall(label_inter, relevance, k)
                elif metric == 'ndcg':
                    result[i, j] += ndcg(label_inter, relevance, k)
                elif metric == 'precision':
                    result[i, j] += precision(relevance, k)
                elif metric == 'mrr':  
                    result[i, j] += mrr(relevance, k) 
    result /= len(test_dataloader.dataset)
    # Log results 
    header, values = '', ''
    for i, metric in enumerate(metrics):
        for j, k in enumerate(top_ks):
            name = f'{metric}@{k}'
            header += f'{name:>16s}'
            values += f'{result[i][j]:>16.4f}'
    logging.info("[TEST]")
    logging.info(header)
    logging.info(values)
    logging.info('')