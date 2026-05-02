import pandas as pd
from psycopg2.pool import SimpleConnectionPool
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from typing import Literal
from utils.get_marked import get_Hatom1, get_inner_ba12, convert_implicit_H

class DatabasePool:
    """Database connection pool management class"""
    def __init__(self, minconn=1, maxconn=5, **kwargs):
        self.pool = SimpleConnectionPool(minconn, maxconn, **kwargs)
    
    @contextmanager
    def get_connection(self):
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

def get_avail_mol_pool(df: pd.DataFrame, columns='fg1_fg2', max_workers=3, type_source='3w2'):
    '''The parallel version using the connection pool'''
    # if columns != 'fg1_fg2':
    #     df.drop_duplicates(columns, inplace=True)
    
    db_pool = DatabasePool( 
        minconn=1,
        maxconn=max_workers,
        database='bide_DB',
        host='localhost',
        port=30825
    )
    
    def get_single(fg1_fg2: str,type_source:Literal['3w2', '463', '581']):
        """Single query function, using connection pool"""
        with db_pool.get_connection() as conn:
            with conn.cursor() as curs:
                if type_source == '3w2':
                    curs.execute(
                        "SELECT smiles FROM dpph.avail_smiles WHERE mol@> %s::qmol ORDER BY ac, amw LIMIT 20", (fg1_fg2,)  #ECFP 3W2 
                    )
                elif type_source == '463':
                    curs.execute(
                        "SELECT smiles FROM dpph.avail_smiles_1044 WHERE mol@> %s::qmol AND is_463=1 ORDER BY ac, amw LIMIT 20", (fg1_fg2,)  #ECFP  
                    )
                elif type_source == '581':
                    curs.execute(
                        "SELECT smiles FROM dpph.avail_smiles_1044 WHERE mol@> %s::qmol AND is_463=0 ORDER BY ac, amw LIMIT 20", (fg1_fg2,)  #ECFP  
                    )
                elif type_source == '3w2_add1044':
                    curs.execute(
                        "SELECT smiles FROM dpph.avail_smiles3w2_1044 WHERE mol@> %s::qmol ORDER BY ac, amw LIMIT 20", (fg1_fg2,)  #ECFP  
                    )
                res = curs.fetchall()
                return pd.DataFrame(res).values.reshape(-1).tolist()
    
    unique_values = df[columns].tolist()
    results = {}

    # parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        from concurrent.futures import as_completed
        future_to_value = {
            executor.submit(get_single, value, type_source=type_source): value 
            for value in unique_values
        }
        
        # collect results
        for future in as_completed(future_to_value):
            value = future_to_value[future]
            try:
                results[value] = future.result()
            except Exception as e:
                print(f"Error occurred while querying {value}: {e}")
                results[value] = []
    
    # Mapping result
    df['avail'] = df[columns].map(results)
    return df





def uniform_format(df_:pd.DataFrame,type_ = 'H_inner', mol_source:Literal['3w2', '1044']='3w2', not_consider_H = True):
    if type_ == 'H_inner':
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
        # if not_consider_H:
        H_inner = get_avail_mol_pool(H_inner, 'fg1_fg2_marked', type_source= mol_source)   #consider implicit H
        # else:
        #     H_inner['smarts_add_H'] = H_inner['smarts_inner_marked'].apply(convert_implicit_H)
        #     H_inner = get_avail_mol_pool(H_inner, 'smarts_add_H',type_source= mol_source)
        #     del H_inner['smarts_add_H']

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

    # #bf of bonds within functional groups
    fgs = pd.read_csv('data/priority_fgs.txt',header=None,delimiter='\t')
    H_inner0 = pd.read_csv('data/H_inner_marked.csv', delimiter='\t')
    inner0 = pd.read_csv('data/inner_marked.csv',delimiter='\t')
    df_fg1_fg20 = pd.read_csv('data/type4_construct_fg_fg.csv',delimiter='\t')
    avail_1044 = pd.read_csv('data/avail_smiles_3w2.csv')

    save_path = 'result_avail_mols'
    for source in ['3w2', '463', '581']:
        if os.path.exists(f'{save_path}/combined_df_{source}.csv'):
            break
        #add H FG info
        H_inner = uniform_format(H_inner0.copy(), type_ = 'H_inner', mol_source=source)
        H_inner.to_csv(f'{save_path}/dpph_Hinner_matched_{source}.csv', index=False)

        #add inner FG info
        inner = uniform_format(inner0.copy(), type_ = 'inner', mol_source=source )
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
        combined_df = pd.concat([df_fg1_fg2, inner, H_inner])
        combined_df['mol_source'] = source
        combined_df.to_csv(f'{save_path}/combined_df_{source}.csv',index=False)
    
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

    for type_ in ['fg1_fg2','Hinner','inner']:
        mapping = combined_df_1044.loc[combined_df_1044['type'] == type_].set_index('fg1_fg2_marked')['avail0']
        
        mask = (combined_df_3w2['type'] == type_) & combined_df_3w2['is_1044'] #combined_df_3w2.loc[]
        combined_df_3w2.loc[mask, 'avail0'] = combined_df_3w2.loc[mask, 'fg1_fg2_marked'].map(mapping)

    combined_df_3w2 = combined_df_3w2.sort_values('is_1044',ascending=False).drop_duplicates(subset='fg1_fg2_marked',keep='first')
    combined_df_3w2.drop(columns=['is_463'],inplace=True)
    combined_df_3w2.drop_duplicates(subset='canon_smarts').to_csv('result/represent_fg1_fg2.csv',index=False) #
    combined_df_3w2[combined_df_3w2['is_1044']].drop_duplicates(subset='canon_smarts').to_csv('result/represent_fg1_fg2_1044.csv',index=False)

    print('The bond of 3w2 and 1044 are combined completed !')
