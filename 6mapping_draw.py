import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pickle,os



df_all_bond = pd.concat([pd.read_pickle(f'{x}_add_fp.pickle')for x in ['fg1_fg2','inner','H_inner']])#combined all bond
avail_bond = pd.read_csv('result/represent_fg1_fg2.csv') 
avail_bond_1044 = avail_bond.loc[avail_bond['is_1044']]  #1044
# exit(0)



subset1_idx = df_all_bond.loc[df_all_bond['fg1_fg2_marked'].isin(avail_bond['fg1_fg2_marked'])].index      # bule 
subset2_idx = df_all_bond.loc[df_all_bond['fg1_fg2_marked'].isin(avail_bond_1044['fg1_fg2_marked'])].index # orange

print(f"all bond: {len(df_all_bond)}")
print(f"subset1: {len(subset1_idx)}")
print(f"subset2: {len(subset2_idx)}")


if not os.path.exists('X_2d.pkl'):
    X = np.vstack(df_all_bond['fp'].values)

    pca = PCA(n_components=2)
    print('starting pca...')
    X_2d = pca.fit_transform(X)

    pickle.dump(X_2d, open('X_2d.pkl','wb'))
else:
    X_2d = pickle.load(open('X_2d.pkl','rb'))


# ==================== 3. draw ====================
plt.figure(figsize=(10, 8))

plt.scatter(X_2d[:, 0], X_2d[:, 1],
            c='gray', alpha=0.3, s=20, label='all bonds', zorder=1)

plt.scatter(X_2d[subset1_idx, 0], X_2d[subset1_idx, 1],
            c='dodgerblue', alpha=0.8, s=20, label='available bonds', edgecolors='k', linewidth=0.5, zorder=2)

plt.scatter(X_2d[subset2_idx, 0], X_2d[subset2_idx, 1],
            c='darkorange', alpha=0.8, s=20, label='initiated bonds', edgecolors='k', linewidth=0.5, zorder=3)

plt.title('PCA', fontsize=14)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)

plt.legend(markerscale=1.5, fontsize=11, framealpha=0.9)

# add grid
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim((-4, 6.3))
plt.ylim((-4, 5))

plt.tight_layout()
plt.savefig('fingerprint_2d_visualization.png', dpi=300, bbox_inches='tight')


# plt.show()