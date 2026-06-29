import os
import sys
import argparse
import torch
import torch.nn.functional as F
import shutil
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for _path in (_PROJECT_ROOT, _SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from protgnn_analysis.models import GnnNets # GnnNets_NC removed
from protgnn_analysis.load_dataset import get_dataset, get_dataloader
from protgnn_analysis.config import data_args, train_args, model_args
from protgnn_analysis.my_mcts import mcts
from tqdm import tqdm

def _compute_class_weights(labels, num_classes):
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    nonzero_mask = class_counts > 0
    class_weights = np.ones(num_classes, dtype=np.float32)
    class_weights[nonzero_mask] = len(labels) / (num_classes * class_counts[nonzero_mask])
    return torch.tensor(class_weights, dtype=torch.float32, device=model_args.device), class_counts

def _average_precision_score(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    positive_count = y_true.sum()
    if positive_count == 0: return float('nan')
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positive_count
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))

def _collect_binary_metrics(labels, predictions, positive_scores):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    ap = _average_precision_score(labels, positive_scores)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'pr_auc': ap}

def _format_eval_metrics(split_name, metrics):
    msg = f"{split_name} | Loss: {metrics['loss']:.3f} | Acc: {metrics['acc']:.3f}"
    if 'pr_auc' in metrics:
        msg += f" | PR-AUC: {metrics['pr_auc']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}"
    return msg

def warm_only(model):
    for p in model.model.gnn_layers.parameters(): p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters(): p.requires_grad = False

def joint(model):
    for p in model.parameters(): p.requires_grad = True

def append_record(info):
    with open('./log/hyper_search', 'a') as f:
        f.write(info + '\n')

def train_GC(clst, sep):
    print(f"Weights: Cluster={clst}, Separation={sep}")
    print('start loading data====================')
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name)
    input_dim = dataset.num_node_features # Should be 3 now
    output_dim = int(dataset.num_classes)
    
    dataloader = get_dataloader(
        dataset, train_args.batch_size,
        random_split_flag=data_args.random_split,
        data_split_ratio=data_args.data_split_ratio,
        seed=data_args.seed
    )

    print('start training model==================')
    gnnNets = GnnNets(input_dim, output_dim, model_args)
    gnnNets.to_device()
    
    # Handle Class Imbalance (HOME vs ADMITTED)
    train_indices = dataloader['train'].dataset.indices
    train_labels = np.array([int(dataset[idx].y.item()) for idx in train_indices])
    class_weights, class_counts = _compute_class_weights(train_labels, output_dim)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    # Logging stats
    avg_nodes = sum([d.num_nodes for d in dataset]) / len(dataset)
    print(f"Graphs: {len(dataset)}, Avg Nodes: {avg_nodes:.2f}, Classes: {output_dim}")
    print(f"Train Class Counts: {class_counts.astype(int).tolist()}")

    ckpt_dir = os.path.join(model_args.checkpoint, data_args.dataset_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs('./log', exist_ok=True)

    best_metric = -1.0
    early_stop_count = 0

    for epoch in range(train_args.max_epochs):
        acc, loss_list = [], []
        
        # Prototype projection
        if epoch >= train_args.proj_epochs and epoch % 10 == 0:
            gnnNets.eval()
            print("Projecting prototypes...")
            for i in range(output_dim * model_args.num_prototypes_per_class):
                label = i // model_args.num_prototypes_per_class
                count, best_similarity = 0, 0
                for j in range(len(train_indices)):
                    data = dataset[train_indices[j]]
                    if data.y == label:
                        count += 1
                        _, similarity, prot = mcts(data, gnnNets, gnnNets.model.prototype_vectors[i])
                        if similarity > best_similarity:
                            best_similarity = similarity
                            proj_prot = prot
                    if count >= 10:
                        gnnNets.model.prototype_vectors.data[i] = proj_prot
                        break

        gnnNets.train()
        if epoch < train_args.warm_epochs: warm_only(gnnNets)
        else: joint(gnnNets)

        for batch in dataloader['train']:
            logits, probs, _, _, min_distances = gnnNets(batch)
            loss = criterion(logits, batch.y.view(-1))
            
            # Prototype Losses
            prot_correct = torch.t(gnnNets.model.prototype_class_identity[:, batch.y].bool()).to(model_args.device)
            cluster_cost = torch.mean(torch.min(min_distances[prot_correct].reshape(-1, model_args.num_prototypes_per_class), dim=1)[0])
            separation_cost = -torch.mean(torch.min(min_distances[~prot_correct].reshape(-1, (output_dim-1)*model_args.num_prototypes_per_class), dim=1)[0])
            
            loss = loss + clst*cluster_cost + sep*separation_cost

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        # Evaluation
        eval_state = evaluate_GC(dataloader['eval'], gnnNets, criterion)
        print(f"Epoch {epoch:03d} | Train Loss: {np.mean(loss_list):.3f} | {_format_eval_metrics('Eval', eval_state)}")
        
        # Save Best
        sel_metric = eval_state.get('pr_auc', eval_state['acc'])
        if sel_metric > best_metric:
            best_metric = sel_metric
            early_stop_count = 0
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, sel_metric, True)
        else:
            early_stop_count += 1
            if early_stop_count > train_args.early_stopping: break

    # Final Test
    checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best.pth'))
    gnnNets.update_state_dict(checkpoint['net'])
    test_state = evaluate_GC(dataloader['test'], gnnNets, criterion)
    print(f"FINAL TEST | {_format_eval_metrics('Test', test_state)}")

def evaluate_GC(dataloader, gnnNets, criterion):
    acc, loss_list, labels, probs = [], [], [], []
    gnnNets.eval()
    with torch.no_grad():
        for batch in dataloader:
            logits, p, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y.view(-1))
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())
            labels.append(batch.y.cpu())
            probs.append(p.cpu())

    eval_state = {'loss': np.mean(loss_list), 'acc': np.concatenate(acc).mean()}
    y_true = torch.cat(labels).numpy()
    y_prob = torch.cat(probs).numpy()
    if y_prob.shape[1] == 2:
        eval_state.update(_collect_binary_metrics(y_true, y_prob.argmax(1), y_prob[:, 1]))
    return eval_state

def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    gnnNets.to('cpu')
    state = {'net': gnnNets.state_dict(), 'epoch': epoch, 'acc': eval_acc}
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")
    torch.save(state, ckpt_path)
    gnnNets.to(model_args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clst', type=float, default=0.1)
    parser.add_argument('--sep', type=float, default=0.01)
    args = parser.parse_args()
    train_GC(args.clst, args.sep)
