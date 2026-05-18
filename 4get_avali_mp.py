import pandas as pd
import psycopg2
from psycopg2.pool import SimpleConnectionPool,ThreadedConnectionPool
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from rdkit import Chem
from psycopg2 import sql
import os
import time
from tqdm import tqdm
from typing import Literal
from utils.get_marked import get_Hatom1, get_inner_ba12, convert_implicit_H, convert_H0
from utils.match_lonely import match_with_lone_pair

class DatabasePool:
    """数据库连接池管理类"""
    def __init__(self, minconn=1, maxconn=5, **kwargs):
        # self.pool = SimpleConnectionPool(minconn, maxconn, **kwargs)
        self.pool = ThreadedConnectionPool(minconn, maxconn, **kwargs)
    
    @contextmanager
    def get_connection(self):
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

def get_avail_mol_pool(df: pd.DataFrame, columns='fg1_fg2', max_workers=3,
                       type_source='3w2', batch_size=500):
    '''使用连接池的并行批量查询版本

    优化：UNNEST 数组 + CROSS JOIN LATERAL 将大量单行查询合并为批量查询，
    大幅减少数据库往返次数。batch_size 控制每批处理的 SMARTS 数量。
    '''

    db_pool = DatabasePool(
        minconn=3,
        maxconn=max_workers,
        database='bide_DB',
        host='localhost',
        port=30825
    )

    TABLE_CONFIG = {
        '3w2':          ('dpph.avail_smiles', ''),
        '463':          ('dpph.avail_smiles_1044', 'AND is_463=1'),
        '581':          ('dpph.avail_smiles_1044', 'AND is_463=0'),
        '3w2_add1044':  ('dpph.avail_smiles3w2_1044', ''),
    }
    table_name, extra_cond = TABLE_CONFIG[type_source]

    unique_values = df[columns].unique().tolist()
    if not unique_values:
        df['avail'] = [[] for _ in range(len(df))]
        return df

    results = {}

    def process_batch(batch):
        """处理一批 SMARTS，返回 {smarts: [smiles...]}"""
        batch_results = {}
        try:
            with db_pool.get_connection() as conn:
                with conn.cursor() as curs:
                    # 利用 unnest 数组传参 + LATERAL 实现批量查询
                    # 每个 pattern 独立 LIMIT 20，与单条查询执行计划等效
                    query = f"""
                        SELECT p.smarts, sub.smiles
                        FROM unnest(%s::text[]) WITH ORDINALITY AS p(smarts, id)
                        CROSS JOIN LATERAL (
                            SELECT smiles
                            FROM {table_name} av
                            WHERE av.mol @> p.smarts::qmol
                            {extra_cond}
                            ORDER BY av.ac, av.amw
                            LIMIT 20
                        ) sub
                        ORDER BY p.id
                    """
                    curs.execute(query, (batch,))

                    # 按 smarts 分组收集结果
                    current_smarts = None
                    current_list = None
                    for smarts, smiles in curs.fetchall():
                        if smarts != current_smarts:
                            if current_smarts is not None:
                                batch_results[current_smarts] = current_list
                            current_smarts = smarts
                            current_list = [smiles]
                        else:
                            current_list.append(smiles)
                    if current_smarts is not None:
                        batch_results[current_smarts] = current_list
        except Exception as e:
            print(f"批次查询出错: {e}")
        return batch_results

    # 分批处理
    batches = [unique_values[i:i+batch_size]
               for i in range(0, len(unique_values), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, b) for b in batches]
        for future in futures:
            try:
                results.update(future.result())
            except Exception as e:
                print(f"批次处理出错: {e}")

    # 补齐无匹配项的 SMARTS
    for v in unique_values:
        results.setdefault(v, [])

    df['avail'] = df[columns].map(results)
    return df





def uniform_format(df_:pd.DataFrame, type_ = 'H_inner', mol_source:Literal['3w2', '1044']='3w2', not_consider_H = True):
    if type_ == 'ELE': #smarts	smarts_marked	smarts_marked_oxygen
        ELE_inner = df_.copy()
        ELE_inner['fg1'] = ELE_inner['smarts']
        ELE_inner['fg2'] = 'E'
        ELE_inner['fg1_fg2'] = ELE_inner['smarts']
        ELE_inner['fg1_fg2_marked'] = ELE_inner['smarts_marked']
        ELE_inner['bond'] = 'SINGLE'
        ELE_inner['atom1'] = ELE_inner['smarts_marked'].apply(lambda x: get_Hatom1(x, m_tag=3))
        ELE_inner['atom2'] = 'E'
        ELE_inner['fg1_fg2_marked'] = ELE_inner['smarts_marked']
        ELE_inner['canon_smarts'] = ELE_inner['smarts_marked_oxygen']
        #不匹配H 并且 还有其他条件吗? 有 如：CS(=O)(=O)C
        # match_with_lone_pair 后续矫正
        ELE_inner['smarts_add_H0'] = ELE_inner['smarts_marked'].apply(convert_H0) #排除含有H的SMILES
        ELE_inner = get_avail_mol_pool(ELE_inner, 'smarts_add_H0',type_source= mol_source) 
        del ELE_inner['smarts_add_H0']
        
        #进一步处理
        ELE_inner = ELE_inner.loc[ELE_inner['avail'].astype(bool)]
        ELE_inner_ex = ELE_inner.explode('avail')
        
        ELE_inner_ex['mol'] = ELE_inner_ex['avail'].apply(Chem.MolFromSmiles)
        ELE_inner_ex['lonely'] = ELE_inner_ex.apply(lambda row:match_with_lone_pair(row['mol'], row['fg1_fg2_marked']), axis=1)
        ELE_inner_ex = ELE_inner_ex.loc[ELE_inner_ex['lonely'].astype(bool)]
        ELE_inner_ex = ELE_inner_ex.groupby(by='fg1_fg2_marked')['avail'].apply(list).reset_index()
        #合并
        ELE_inner.drop(columns=['avail'],inplace=True)
        ELE_inner = ELE_inner.merge(ELE_inner_ex,on='fg1_fg2_marked',how='right') 
        ELE_inner = ELE_inner[['fg1', 'fg2', 'fg1_fg2', 'fg1_fg2_marked', 'bond', 'atom1', 'atom2', 'canon_smarts','avail']]
        ELE_inner['type'] = 'ELE'
        ELE_inner = ELE_inner.loc[ELE_inner['avail'].astype(bool)]
        ELE_inner['avail0'] = ELE_inner['avail'].apply(lambda x: x[0])
        
        return ELE_inner
    elif type_ == 'H_inner':
        H_inner = df_.copy()
        H_inner['fg1'] = H_inner['smarts']
        H_inner['fg2'] = 'H'
        H_inner['fg1_fg2'] = H_inner['smarts']
        H_inner['fg1_fg2_marked'] = H_inner['smarts_inner_marked']
        H_inner['bond'] = 'SINGLE'
        H_inner['atom1'] = H_inner['smarts_inner_marked'].apply(get_Hatom1)
        H_inner['atom2'] = 'H'
        H_inner['fg1_fg2_marked'] = H_inner['smarts_inner_marked']
        H_inner['canon_smarts'] = H_inner['smarts_marked_oxygen']
        if not_consider_H:
            H_inner = get_avail_mol_pool(H_inner, 'fg1_fg2_marked', type_source= mol_source) 
        else:
            H_inner['smarts_add_H'] = H_inner['smarts_inner_marked'].apply(convert_implicit_H)
            H_inner = get_avail_mol_pool(H_inner, 'smarts_add_H',type_source= mol_source) #考虑转换为隐式氢
            del H_inner['smarts_add_H']

        H_inner = H_inner[['fg1', 'fg2', 'fg1_fg2', 'fg1_fg2_marked', 'bond', 'atom1', 'atom2', 'canon_smarts','avail']]
        H_inner['type'] = 'Hinner'
        H_inner = H_inner.loc[H_inner['avail'].astype(bool)]
        H_inner['avail0'] = H_inner['avail'].apply(lambda x:x[0])
        return H_inner
    elif type_ == 'inner':
        inner = df_.copy()
        inner['fg1'] = inner['smarts']
        inner['fg2'] = inner['smarts']
        inner['fg1_fg2'] = inner['smarts']
        inner['fg1_fg2_marked'] = inner['smarts_marked']
        inner[['bond','atom1','atom2']] = inner.apply(get_inner_ba12,axis=1, result_type='expand')
        inner['fg1_fg2_marked'] = inner['smarts_marked']
        inner['canon_smarts'] = inner['smarts_marked_oxygen']
        inner = get_avail_mol_pool(inner, 'smarts',type_source=mol_source)

        inner = inner[['fg1', 'fg2', 'fg1_fg2', 'fg1_fg2_marked', 'bond', 'atom1', 'atom2', 'canon_smarts','avail']]
        inner['type'] = 'inner'
        inner = inner.loc[inner['avail'].astype(bool)]
        inner['avail0'] = inner['avail'].apply(lambda x:x[0])
        return inner

if __name__ == '__main__':
    # #官能团内断成键
    fgs = pd.read_csv('data/priority_fgs.txt',header=None,delimiter='\t')
    H_inner0 = pd.read_csv('data/H_inner_marked.csv', delimiter='\t')
    inner0 = pd.read_csv('data/inner_marked.csv',delimiter='\t')
    df_fg1_fg20 = pd.read_csv('data/type4_construct_fg_fg.csv',delimiter='\t')
    ele_lonely0 = pd.read_csv('data/ele_marked.csv', delimiter='\t') #孤对电子

    save_path = 'result_avail_mols'
    for source in tqdm(['3w2', '463', '581']):
        if os.path.exists(f'{save_path}/combined_df_{source}.csv'):
            break
        ele_lonely = uniform_format(ele_lonely0.copy(), type_ = 'ELE', mol_source=source, not_consider_H=False) 
        ele_lonely.to_csv(f'{save_path}/dpph_ele_lonely_matched_{source}.csv', index=False)

        #add H FG info
        H_inner = uniform_format(H_inner0.copy(), type_ = 'H_inner', mol_source=source,not_consider_H=False) #.drop_duplicates(subset='smarts_marked_oxygen')
        H_inner.to_csv(f'{save_path}/dpph_Hinner_matched_{source}.csv', index=False)
        
        #add inner FG info
        inner = uniform_format(inner0.copy(), type_ = 'inner', mol_source=source ) #.drop_duplicates(subset='smarts_marked_oxygen')
        inner.to_csv(f'{save_path}/dpph_inner_matched_{source}.csv', index=False)

        hinner_consider_H = uniform_format(H_inner0.copy(), type_ = 'H_inner', mol_source=source,not_consider_H=True) #不考虑氢
        #outer FG info 
        inner_values = set(hinner_consider_H['fg1']) | set(fgs[0]) - set(H_inner0['smarts']) #.union(set(inner['fg2']))fg1 equal to fg2 #contain single atom
        print('fg1:',len(inner_values))
        df_fg1_fg2 = df_fg1_fg20.copy()
        df_fg1_fg2 = df_fg1_fg2[df_fg1_fg2['fg1'].isin(inner_values) & df_fg1_fg2['fg2'].isin(inner_values)]
        # if False:
        df_fg1_fg2 = get_avail_mol_pool(df_fg1_fg2, columns= 'fg1_fg2', max_workers=3,type_source=source)
        df_fg1_fg2 = df_fg1_fg2.loc[df_fg1_fg2['avail'].astype(bool)]
        df_fg1_fg2['type'] = 'fg1_fg2'
        df_fg1_fg2['avail0'] = df_fg1_fg2['avail'].apply(lambda x:x[0])
        df_fg1_fg2.to_csv(f'{save_path}/dpph_match_fg_fg_{source}.csv', index=False)

        #  combined
        df_fg1_fg2 = pd.read_csv(f'{save_path}/dpph_match_fg_fg_{source}.csv')
        inner = pd.read_csv(f'{save_path}/dpph_inner_matched_{source}.csv')
        H_inner = pd.read_csv(f'{save_path}/dpph_Hinner_matched_{source}.csv')
        ele_lonely = pd.read_csv(f'{save_path}/dpph_ele_lonely_matched_{source}.csv')
        combined_df = pd.concat([df_fg1_fg2, inner, H_inner, ele_lonely]) #, ele_lonely
        # combined_df = pd.concat([ inner, H_inner]) 
        combined_df['mol_source'] = source
        combined_df.to_csv(f'{save_path}/combined_df_{source}.csv',index=False)
    #优先
    #键的合并
    combined_df_3w2 = pd.read_csv(f'{save_path}/combined_df_3w2.csv')
    combined_df_463 = pd.read_csv(f'{save_path}/combined_df_463.csv')
    combined_df_581 = pd.read_csv(f'{save_path}/combined_df_581.csv')
    # combined_df_1044 = pd.read_csv(f'{save_path}/combined_df_1044.csv')
    combined_df_463['is_463']=1
    combined_df_581['is_463']=0

    combined_df_1044 = pd.concat([combined_df_581, combined_df_463])
    combined_df_1044 = combined_df_1044.sort_values('is_463',ascending=False).drop_duplicates(subset='fg1_fg2_marked',keep='first')
    combined_df_3w2 = pd.concat([combined_df_3w2, combined_df_463, combined_df_581])
    combined_df_3w2.reset_index(drop=True, inplace=True)

    combined_df_3w2['is_1044'] = combined_df_3w2['fg1_fg2_marked'].isin(combined_df_1044['fg1_fg2_marked'])
    # combined_df_3w2 = combined_df_3w2.df.sort_values('is_1044', ascending=False).drop_duplicates(subset='is_1044', keep='first')
    for type_ in ['fg1_fg2','Hinner','inner','ELE']:
        mapping = combined_df_1044.loc[combined_df_1044['type'] == type_].set_index('fg1_fg2_marked')['avail0']
        
        mask = (combined_df_3w2['type'] == type_) & combined_df_3w2['is_1044'] #combined_df_3w2.loc[]
        combined_df_3w2.loc[mask, 'avail0'] = combined_df_3w2.loc[mask, 'fg1_fg2_marked'].map(mapping)

    combined_df_3w2 = combined_df_3w2.sort_values('is_1044',ascending=False).drop_duplicates(subset='fg1_fg2_marked',keep='first')
    combined_df_3w2.drop(columns=['is_463'],inplace=True)
    combined_df_3w2.drop_duplicates(subset=['canon_smarts','type']).to_csv('result2/represent_fg1_fg2_add_ele.csv',index=False) #
    combined_df_3w2[combined_df_3w2['is_1044']].drop_duplicates(subset=['canon_smarts','type']).to_csv('result2/represent_fg1_fg2_1044_add_ele.csv',index=False)

    print('The bond of 3w2 and 1044 are combined completed !')