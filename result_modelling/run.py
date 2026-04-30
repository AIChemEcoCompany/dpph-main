import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    X = np.load(open('X.npy','rb'),allow_pickle=True)
    y = np.load(open('y.npy','rb'),allow_pickle=True)

    print(f"feature shape: {X.shape}, Number of labels: {len(y)}")
    print(f"label distribution: negative={sum(y==0)}, positive={sum(y==1)}")

    metrics = ['accuracy', 'precision',  'auc'] #'recall', 'f1',
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),   # probability=True 用于计算AUC
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'auc': make_scorer(roc_auc_score, needs_proba=True)   # 需要概率预测
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results_summary = {}

    print("\n" + "="*60)
    # print("开始5折交叉验证评估")
    print("="*60)


    for name, clf in models.items():
        print(f"\n>>> model: {name}")
        cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring, 
                                    return_train_score=False, verbose=0,n_jobs=16)
        
        summary = {}
        for metric in metrics:
            scores = cv_results[f'test_{metric}']
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            summary[metric.capitalize()] = f"{mean_score:.4f} ± {std_score:.4f}"
            # 同时保留数值用于后续绘图
            summary[f'{metric}_mean'] = mean_score
            summary[f'{metric}_std'] = std_score
        
        results_summary[name] = summary
        
        # # 打印详细结果
        # print(f"准确率 (Accuracy):   {summary['Accuracy']}")
        # print(f"精确率 (Precision):  {summary['Precision']}")
        # print(f"召回率 (Recall):     {summary['Recall']}")
        # print(f"F1分数 (F1-score):   {summary['F1']}")
        # print(f"AUC:                 {summary['Auc']}")

    # ---------- 整理为DataFrame展示 ----------
    # 提取各模型的平均指标（不带标准差）用于绘图
    comparison_df = pd.DataFrame({
        name: {metric: float(results_summary[name][f'{metric.lower()}_mean']) 
            for metric in metrics}
        for name in models.keys()
    }).T

    print("\n" + "="*60)
    print("5折交叉验证平均性能汇总")
    print("="*60)
    print(comparison_df.round(4))

    # ---------- 可视化对比 ----------


    comparison_df.plot(kind='bar', figsize=(12,6), rot=0)
    plt.title('Model Performance Comparison (5-fold Cross-Validation)')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'models_result.png')
    # plt.show()

    # 可选：同时绘制带误差条的图（展示标准差）
    # 准备数据
    # metrics_list = metrics#['Accuracy', 'Precision', 'Recall', 'F1', 'Auc']
    # fig, axes = plt.subplots(1, len(metrics_list), figsize=(20, 5))
    # for i, metric in enumerate(metrics_list):
    #     means = [results_summary[name][f'{metric.lower()}_mean'] for name in models.keys()]
    #     stds = [results_summary[name][f'{metric.lower()}_std'] for name in models.keys()]
    #     axes[i].bar(models.keys(), means, yerr=stds, capsize=5, color='steelblue')
    #     axes[i].set_title(metric)
    #     axes[i].set_ylim(0, 1)
    #     axes[i].tick_params(axis='x', rotation=45)
    #     axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('res.png')