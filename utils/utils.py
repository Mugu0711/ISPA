# utils/utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import _tree, export_text
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def save_fig(fig, path, dpi=150):
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

def plot_hist_box(df, col, outpath=None):
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    sns.histplot(df[col].dropna(), ax=axes[0])
    sns.boxplot(x=df[col].dropna(), ax=axes[1])
    axes[0].set_title(f'Histogram of {col}')
    axes[1].set_title(f'Boxplot of {col}')
    if outpath:
        save_fig(fig, outpath)
    else:
        plt.show()

def plot_bar_percent(df, cat_col, target='Churn', outpath=None):
    # percent churn by category
    tab = df.groupby(cat_col)[target].apply(lambda x: (x=='Yes').mean()).reset_index()
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=tab, x=cat_col, y=target, ax=ax)
    ax.set_ylabel('Churn rate (fraction)')
    ax.set_title(f'Churn rate by {cat_col}')
    plt.xticks(rotation=45)
    if outpath:
        save_fig(fig, outpath)
    else:
        plt.show()

def compute_lift_gains(y_true, y_proba, n_bins=10):
    """
    Returns DataFrame with decile, count, churn_count, churn_rate, lift, cum_captured
    """
    df = pd.DataFrame({'y': y_true, 'proba': y_proba})
    df = df.sort_values('proba', ascending=False).reset_index(drop=True)
    df['decile'] = pd.qcut(df.index, q=n_bins, labels=False) + 1
    grouped = df.groupby('decile').agg(count=('y','size'), churn_count=('y', lambda x: (x=='Yes').sum()))
    grouped['churn_rate'] = grouped['churn_count'] / grouped['count']
    base_rate = (df['y']=='Yes').mean()
    grouped['lift'] = grouped['churn_rate'] / base_rate
    grouped['cum_churn'] = grouped['churn_count'].cumsum()
    grouped['cum_captured'] = grouped['cum_churn'] / df['y'].eq('Yes').sum()
    return grouped.sort_index(ascending=True)

def plot_lift(df_lift, outpath=None):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(df_lift.index, df_lift['lift'], marker='o')
    ax.set_xlabel('Decile (1 = top scored)')
    ax.set_ylabel('Lift')
    ax.set_title('Lift by Decile')
    if outpath:
        save_fig(fig, outpath)
    else:
        plt.show()

def plot_gains(df_lift, outpath=None):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(df_lift.index, df_lift['cum_captured'], marker='o')
    ax.set_xlabel('Decile (1 = top scored)')
    ax.set_ylabel('Cumulative capture of churners')
    ax.set_title('Cumulative Gains')
    if outpath:
        save_fig(fig, outpath)
    else:
        plt.show()

def plot_roc(y_true, y_proba, outpath=None):
    y_bin = (y_true=='Yes').astype(int)
    fpr, tpr, _ = roc_curve(y_bin, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0,1],[0,1], linestyle='--', color='grey')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    if outpath:
        save_fig(fig, outpath)
    else:
        plt.show()
    return roc_auc

def plot_precision_recall(y_true, y_proba, outpath=None):
    y_bin = (y_true=='Yes').astype(int)
    precision, recall, _ = precision_recall_curve(y_bin, y_proba)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(recall, precision)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    if outpath:
        save_fig(fig, outpath)
    else:
        plt.show()

def tree_rules_text(clf, feature_names, max_depth=4):
    # returns readable rules using sklearn export_text
    try:
        txt = export_text(clf, feature_names=feature_names, max_depth=max_depth)
        return txt
    except Exception as e:
        return str(e)

def tree_rules_from_tree(clf, feature_names):
    # More detailed traversal producing rule -> value
    tree = clf.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree.feature
    ]

    paths = []

    def recurse(node, path):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree.threshold[node]
            # left
            recurse(tree.children_left[node], path + [f"({name} <= {threshold:.3f})"])
            # right
            recurse(tree.children_right[node], path + [f"({name} > {threshold:.3f})"])
        else:
            value = tree.value[node][0]
            total = value.sum()
            probs = value / total
            paths.append((path, total, probs))
    recurse(0, [])
    lines = []
    for p, total, probs in paths:
        lines.append(f"Rule: {' AND '.join(p)} -> samples={int(total)}, prob(no)={probs[0]:.3f}, prob(yes)={probs[1]:.3f}")
    return "\n".join(lines)
