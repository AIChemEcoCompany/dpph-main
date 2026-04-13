import pandas as pd
import numpy as np
from utils.get_fp import tanimoto_np, smol_to_fp, get_Mfp
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
import os


def calculate_hybrid_score(group, bit_weights):
    """deal each group of (fg1_species, fg2_species) """
    # Separate the reported (K) from the unreported (U)
    known = group[group['broken_freq'] > 0]
    unknown = group[group['broken_freq'] == 0].copy()
    
    if unknown.empty:
        print('unknown empty')
        return unknown

    # A. calc Rarity 
    def compute_rarity(fp):
        if fp is None: 
            print('fp is none', unknown['fg1_fg2_marked'])
            return 0.0
        # on_bits = list(fp.GetOnBits())
        on_bits = np.where(fp)[0]
        if on_bits.sum() == 0:
            print('on_bits_0',unknown['fg1_fg2_marked'])

        return np.sum(bit_weights[on_bits]) / len(on_bits) if on_bits.any() else 0.0

    unknown['rarity_raw'] = unknown['fp'].apply(compute_rarity)
    
    # Rarity normalization (0-1)
    r_max = unknown['rarity_raw'].max()
    r_min = unknown['rarity_raw'].min()
    if r_max > r_min:
        unknown['rarity_norm'] = (unknown['rarity_raw'] - r_min) / (r_max - r_min)
    else:
        unknown['rarity_norm'] = 0.5

    # B. Dual-track judgment
    if not known.empty:
        # case 1: If there are reported data within the same category -> Calculate MaxSim and apply weighting
        known_fps = known['fp']#.tolist()
        max_sims = []
        for unk_fp in unknown['fp']:
            if unk_fp is None:
                print(known['fg1_fg2_marked'])
                # max_sims.append(0.0)
                continue
            else:
                # sims = DataStructs.BulkTanimotoSimilarity(unk_fp, known_fps)# batch calc Tanimoto 
                sims = tanimoto_np(unk_fp, np.vstack(known_fps))
            
            max_sims.append(max(sims))
        
        unknown['max_sim'] = max_sims
        # Integration formula: 0.6 * (1 - MaxSim) + 0.4 * Rarity
        unknown['novelty_score'] = 0.6 * (1 - unknown['max_sim']) + 0.4 * unknown['rarity_norm']
        unknown['score_type'] = 'Hybrid'
    else:
        # case 2: No reported data available in the same category -> Sorted only by Rarity
        unknown['novelty_score'] = unknown['rarity_norm']
        unknown['max_sim'] = np.nan
        unknown['score_type'] = 'Rarity-Only'
    
    return unknown



def run_novelty_pipeline(df,type_='fg1_fg2'):
    """df columns: smarts, broken_freq, fg1_species, fg2_species    """
    # Step 1: Parallel computing fingerprint
    print(f"Step 1: The fingerprint of {type_} data is being calculated...")

    if not os.path.exists(f'{type_}_fp.pkl'):
        fp_results = Parallel(n_jobs=16, backend='loky')(
            delayed(get_Mfp)(smi, type_) for smi in tqdm(df['fg1_fg2_marked'], desc="Fingerprint Conversion")
        )
        pickle.dump(fp_results, open(f'{type_}_fp.pkl','wb'))
    else:
        fp_results = pickle.load(open(f'{type_}_fp.pkl','rb'))
    df['fp'] = fp_results

    if not os.path.exists(f"{type_}_add_fp.pickle"):
        df.to_pickle(f'{type_}_add_fp.pickle')

    # Step 2: Calculate the global Bit weight (Method B: Based on the rarity of data reported across the entire industry)
    print(f"Step 2: Calculate the global structure rarity weight table...")
    known_all = df[df['broken_freq'] > 0]['fp'].values
    print('known_all:', len(known_all))
    print('unknown_all:', len(df[df['broken_freq'] <= 0]))

    bit_counts = np.zeros(2048, dtype=np.int32)
    for fp in tqdm(known_all, desc="Count the known loci"):
        # bit_counts += get_bit_array(fp)
        bit_counts += fp #fp: np.array
    
    # calc bit_weights w = -log2(p)
    probs = (bit_counts + 1) / (len(known_all) + 2)
    bit_weights = -np.log2(probs)

    
    print("Step 3: Perform sorting within the classification group...")
    tqdm.pandas(desc="Similarity calculation within the group")
    # Group and process according to fg1_species and fg2_species
    result_df = df.groupby(['fg1_species', 'fg2_species'], group_keys=False).apply(
        lambda x: calculate_hybrid_score(x, bit_weights)
    )
    print('sort:', len(result_df))

    print(f"Step 4: Complete {type_}. The final descending order sorting is in progress...")
    final_output = result_df.sort_values(
        by=['fg1_species', 'fg2_species', 'novelty_score'], 
        ascending=[True, True, False]
    )
    
    return final_output


if __name__ =='__main__':

    df_fg1_fg2 = pd.read_csv('result/type4_construct_fg_fg_count.csv',delimiter='\t')
    df_fg1_fg2_novelty = run_novelty_pipeline(df_fg1_fg2, 'fg1_fg2')
    df_fg1_fg2_novelty['type']='fg1_fg2'
    print(len(df_fg1_fg2) - len(df_fg1_fg2_novelty) ,'rest!') #96
    # df_fg1_fg2_novelty.to_pickle('sorted_fg_fg.pickle')
    #avail bonds sorted
    # df_fg1_fg2_novelty_avail = df_fg1_fg2_novelty[df_fg1_fg2_novelty['fg1_fg2_marked'].isin(avail_bond['fg1_fg2_marked'])]
    # df_fg1_fg2_novelty.to_csv('sorted_fg1_fg2_avail.csv',index=False)

    H_inner = pd.read_csv('result/H_inner_marked_count.csv',delimiter='\t') 
    H_inner_novelty = run_novelty_pipeline(H_inner, 'H_inner')
    H_inner_novelty['type']='Hinner'
    print(len(H_inner) - len(H_inner_novelty) ,'rest!')#74

    # H_inner_novelty.to_pickle('sorted_H_inner.pickle')
    #avail bonds sorted
    # H_inner_novelty_avail = H_inner_novelty[H_inner_novelty['fg1_fg2_marked'].isin(avail_bond['fg1_fg2_marked'])]
    # H_inner_novelty.to_csv('sorted_H_inner_avail.csv',index=False)

    inner = pd.read_csv('result/inner_marked_count.csv',delimiter='\t')
    inner_novelty = run_novelty_pipeline(inner,'inner')
    inner_novelty['type']='inner'
    print(len(inner) - len(inner_novelty) ,'rest!') #52

    # inner_novelty.to_pickle('sorted_inner.pickle')
    #avail bonds sorted
    # inner_novelty_avail = inner_novelty[inner_novelty['fg1_fg2_marked'].isin(avail_bond['fg1_fg2_marked'])]
    
    #combined
    combined_df_novelty = pd.concat([df_fg1_fg2_novelty,H_inner_novelty, inner_novelty])
    print(len(combined_df_novelty),'rest!')
    combined_df_novelty.drop_duplicates(subset='canon_smarts',inplace=True)
    print('drop_duplicates',len(combined_df_novelty),'rest!')

    combined_df_novelty.drop(columns=['fp', 'broken_freq', 'formed_freq', 'rarity_raw', 'rarity_norm', 'max_sim'],inplace=True)

    avail_bond = pd.read_csv('result/represent_fg1_fg2.csv')
    combined_df_avail = combined_df_novelty[combined_df_novelty['fg1_fg2_marked'].isin(avail_bond['fg1_fg2_marked'])]
    combined_df_novelty.to_csv('result_sorted/sorted_novelty.csv', index=False)
    combined_df_avail.to_csv('result_sorted/sorted_novelty_avail.csv',index=False)

    
