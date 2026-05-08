import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import pickle
from sklearn.base import clone


if __name__ == '__main__':
    X = np.load(open('X.npy','rb'), allow_pickle=True)
    y = np.load(open('y.npy','rb'), allow_pickle=True)
    
    X_test_index = [184335, 184339, 58926, 94770, 137266, 94772, 137269, 94774, 217655, 217656,
                 16023, 41090,  1685,   769,  2433,  5311, 37819, 39188, 17568, 19769] 
    x_test_x = X[X_test_index]
    x_test_y = y[X_test_index]

    mask = np.ones(X.shape[0], dtype=bool)
    mask[X_test_index] = False
    
    X = X[mask]
    y = y[mask]
    

    print(f"Shape of the feature matrix: {X.shape}, Number of tags: {len(y)}")
    print(f"Categorical Distribution: positive={sum(y==0)}, negative={sum(y==1)}")

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=16),
        ## 'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, random_state=42),

        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced',solver='liblinear'),
        # 'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
        'SVM': LinearSVC(random_state=42, class_weight='balanced'),
        'LightGBM': LGBMClassifier(   objective='binary',  n_estimators=100,   num_leaves=31,   learning_rate=0.05,     
                                    is_unbalance=True,            
                                    random_state=42,
                                    n_jobs=16,                     
                                    verbosity=-1                    
                                )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results_summary = {}
    high_conf_pos_indices = {}
    PROB_THRESHOLD = 0.6

    print("\n" + "="*60)
    print("="*60)

    for name, clf in models.items():
        print(f"\n>>> dealing model: {name}")
        
        # 存储每折的验证分数和选中的正例索引
        fold_scores = {m: [] for m in metrics}
        selected_indices_per_fold = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model = clone(clf)
            model.fit(X_train, y_train)
            
            if hasattr(model, "predict_proba"):
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]
            else:  
                y_pred = model.predict(X_val)
                y_proba = model.decision_function(X_val)  
            
 
            fold_scores['accuracy'].append(accuracy_score(y_val, y_pred))
            fold_scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            fold_scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            fold_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            if hasattr(model, "predict_proba"):
                fold_scores['auc'].append(roc_auc_score(y_val, y_proba))
            else:
                fold_scores['auc'].append(np.nan)
            
            
            pos_mask_val = (y_val == 1)
            if pos_mask_val.any():
                pos_proba = y_proba[pos_mask_val]
                high_conf_in_val = np.where(pos_mask_val)[0][pos_proba >= PROB_THRESHOLD]
                global_indices = val_idx[high_conf_in_val]
                selected_indices_per_fold.extend(global_indices.tolist())
        
        # 去重并记录此模型选中的正例索引
        unique_selected = list(set(selected_indices_per_fold))
        high_conf_pos_indices[name] = unique_selected
        
        # 汇总交叉验证性能（平均值±标准差）
        summary = {}
        for m in metrics:
            scores = fold_scores[m]
            mean_score = np.nanmean(scores)   # 处理可能的NaN
            std_score = np.nanstd(scores)
            summary[m.capitalize()] = f"{mean_score:.4f} ± {std_score:.4f}"
            summary[f'{m}_mean'] = mean_score
            summary[f'{m}_std'] = std_score
        results_summary[name] = summary
        
        final_model = clone(clf)
        final_model.fit(X, y)
        with open(f'{name}_final.pkl', 'wb') as f:
            pickle.dump(final_model, f)

    comparison_df = pd.DataFrame({
        name: {m: results_summary[name][f'{m}_mean'] for m in metrics}
        for name in models.keys()
    }).T

    # comparison_df.to_csv('comparison_df.csv')
    print("\n" + "="*60)
    print("5折交叉验证平均性能汇总")
    print("="*60)
    print(comparison_df.round(4))

    comparison_df.plot(kind='bar', figsize=(12,6), rot=0)
    plt.title('Model Performance Comparison (5-fold Cross-Validation)')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.savefig('models_result.png')
    # plt.show()

    #Test Set
    results = []
    for name, clf in models.items():
        with open(f'{name}_final.pkl', 'rb') as f:
            model = pickle.load(f)
        
        y_pred = model.predict(x_test_x)
        y_val = x_test_y
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        if hasattr(model, "predict_proba"):
            roc = roc_auc_score(y_val, model.predict_proba(x_test_x)[:, 1])
        else:
            roc = 0.0
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC AUC': roc
        })
        print(name, acc, prec, rec, f1, roc)

    df = pd.DataFrame(results)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    models = df['Model'].values

    x = np.arange(len(models))         
    width = 0.15                       
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 6))
    for metric in metrics:
        values = df[metric].values
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=metric)
        ax.bar_label(rects, fmt='%.3f', padding=2, fontsize=8)
        multiplier += 1

    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('res.png')
    plt.show()