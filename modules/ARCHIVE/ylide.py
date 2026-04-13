import random
import pandas as pd

random.seed(42)

def get_ylide_smiles(n_samples = 100):
    df = pd.read_csv("data/Ylide/Ylide-20240319-bf-0.csv")

    # 获取所有分子
    smiles_df = df[["底物1", "底物2", "底物3", "底物4", "底物5", "底物6", "产物1", "产物2", "产物3", "产物4", "产物5", "产物6", "产物7"]].drop_duplicates()
    smiles_list = smiles_df.values.flatten()
    # 排除空值
    smiles_list = [x for x in smiles_list if str(x) != 'nan']
    # 去重
    smiles_list = list(set(smiles_list))

    # 从smiles_list中随机取 n_samples 个分子
    result_smiles_list = random.sample(smiles_list, n_samples)

    return result_smiles_list