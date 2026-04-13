from rxnmapper import RXNMapper #torch==2.1.0 torchdata==0.7.0 conflict
# from localmapper import localmapper #torch==2.0.1 torchdata==0.8.0
from rdkit import Chem
# from indigo import *
from collections import defaultdict

def remove_mapping(mapped_smiles):
    mol = Chem.MolFromSmiles(mapped_smiles)
    smiles_without_map = mapped_smiles
    if mol is not None:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        smiles_without_map = Chem.MolToSmiles(mol, allHsExplicit=True)
    return smiles_without_map

# def data_process(reactions):
#     rxns = []
#     rxn_mapper = RXNMapper()
#     for reaction in reactions:
#         temp = ''
#         for compound in reaction:
#             if compound['role'] != 'product':
#                 if len(temp) == 0:
#                     temp = compound['smiles']
#                 else:
#                     temp = temp + '.' + compound['smiles']
#             elif ">>" not in temp:
#                 temp = temp + ">>" + compound['smiles']
#             else:
#                 temp = temp + '.' + compound['smiles']
#         rxns.append(temp)
#     try:
#         results = rxn_mapper.get_attention_guided_atom_maps(rxns)
#     except:
#         return 400
#     for i in range(len(results)):
#         smarts = results[i]['mapped_rxn']
#         substrates, products = results[i]['mapped_rxn'].split('>>')[0], results[i]['mapped_rxn'].split('>>')[1]
#         all_smiles = substrates.split('.') + products.split('.')
#         for j in range(len(reactions[i])):
#             for k in range(len(reactions[i])):
#             #print(reactions)
#                 t = remove_mapping(all_smiles[k])
#                 if check(reactions[i][j]['smiles'], t):
#                     reactions[i][j]['mapped_smiles'] = all_smiles[k]
#         reactions[i].append({'confidence': results[i]['confidence']})
#     return reactions

def data_process(reactions, mapper:str='RXN'):#rxn_mapper:localmapper
    rxns = []
    for reaction in reactions:
        temp = ''
        for compound in reaction:
            if compound['role'] != 'product':
                if len(temp) == 0:
                    temp = compound['smiles']
                else:
                    temp = temp + '.' + compound['smiles']
            elif ">>" not in temp:
                temp = temp + ">>" + compound['smiles']
            else:
                temp = temp + '.' + compound['smiles']
        rxns.append(temp)
    
    final_result = []
    if mapper == 'local':
        local_mapper = localmapper('cpu')
        mapped_rxn = local_mapper.get_atom_map(rxns[0])
        mapping_confidence = 0.6
    elif mapper == 'indigo':
        indigo = Indigo()
        indigo_mapper = indigo.loadReaction(rxns[0])
        indigo_mapper.automap("discard")
        mapped_rxn  = indigo_mapper.smiles()
        r_, p_ = mapped_rxn.split(">>")
        r_, p_ = r_.split("."), p_.split(".") 
        #
        mol_ps = [Chem.MolFromSmiles(mol_p) for mol_p in p_]
        for mol_p in mol_ps:
            mapNums = defaultdict(int)
            for atom in mol_p.GetAtoms():
                mapNums[atom.GetAtomMapNum()] += 1
            for atom in mol_p.GetAtoms():
                if mapNums[atom.GetAtomMapNum()] >= 2:
                    # mapNums[atom.GetAtomMapNum()] -= 1 
                    atom.SetAtomMapNum(0)
        mapping_confidence = 0.5
    # rxn_mapper = localmapper() 1
    elif mapper == 'RXN':
        rxn_mapper = RXNMapper()
        try:
            #rxn_mapper.get_attention_guided_atom_maps(rxns=res) 
            result = rxn_mapper.get_attention_guided_atom_maps(rxns) 
            # result = rxn_mapper.get_atom_map(rxn)
        except:
            return [], 0.
        mapped_rxn = result[0]["mapped_rxn"] # 4 result[0]
        # mapped_rxn = result#local map? RXN mapper?
        mapping_confidence:float = result[0]['confidence'] 

    if ">>" in mapped_rxn:
        subs = mapped_rxn.split(">>")[0].split(".")
        prods = mapped_rxn.split(">>")[1].split(".")
    this_rxn_result_dicts = []
    for sub in subs:
        this_rxn_result_dicts.append({
            "role": "substrate",
            "mapped_smiles": sub
        })
    for prod in prods:
        this_rxn_result_dicts.append({
            "role": "product",
            "mapped_smiles": prod
        })
        final_result.append(this_rxn_result_dicts)
        return final_result, mapping_confidence 


if __name__ == '__main__':
    res = data_process('CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F','indigo')
    print(res)