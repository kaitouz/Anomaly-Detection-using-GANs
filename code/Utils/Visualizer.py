
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def plot_distribution(scores_neg, scores_pos, number_of_bins):
    counts_neg, bins_neg = np.histogram(scores_neg, bins=number_of_bins)
    counts_pos, bins_pos = np.histogram(scores_pos, bins=number_of_bins)

    fig, ax = plt.subplots()

    hist1 = ax.hist(bins_neg[:-1], bins_neg, weights=counts_neg, alpha=0.5, label='Normal')
    hist2 = ax.hist(bins_pos[:-1], bins_pos, weights=counts_pos, alpha=0.5, label='Abnormal')
    ax.legend(prop={'size': 10})

    plt.show()

def plot_image(imgs):
    normalized_imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
    plt.imshow(np.transpose(normalized_imgs, (1, 2, 0)))
    plt.show()

from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score

def cal_best_threshold(labels, scores):
    precs, recs, thrs = precision_recall_curve(labels, scores)
    f1s = 2 * precs * recs / (precs + recs)
    f2s = 5 * precs * recs / (4 * precs + recs)
    f1s = f1s[:-1]
    f2s = f2s[:-1]
    thrs = thrs[~np.isnan(f2s)]
    f1s = f1s[~np.isnan(f1s)]
    f2s = f2s[~np.isnan(f2s)]
    best_thre = thrs[np.argmax(f2s)]
    best_F1_score = np.max(f1s)
    best_F2_score = np.max(f2s)
    best_predictions = [1 if i > best_thre else 0 for i in scores]
    best_accuracy = accuracy_score(labels, best_predictions)
    
    # print("auc: ", roc_auc_score(labels, scores))
    # print("best_accuracy: ", best_accuracy)
    # print("best_thre: ", best_thre)
    # print("best_F1_score: ", best_F1_score)
    # print("best_F1_score: ", best_F2_score)
    return best_thre, best_F1_score, best_F2_score, best_accuracy