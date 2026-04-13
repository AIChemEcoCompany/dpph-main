# copy smiles data to postgresql 
create datatase nida;
create extension rdkit;
create schema dpph;
create table dpph.avail_smiles(smiles TEXT,cansmiles TEXT, price float, source TEXT, mol mol, ac smallint, amw float);
create table dpph.avail_smiles_1044(smiles TEXT, ac smallint, amw float, mol mol, is_463 smallint);
\copy dpph.avail_smiles(smiles, cansmiles, price, source,  ac, amw) from '/PATH/data' CSV HEADER;
alter table dpph.avail_smiles set mol = mol_from_smiles(smiles::cstring);
\copy dpph.avail_smiles(smiles,  mol, ac, amw) from '/PATH/data' CSV HEADER;
alter table dpph.avail_smiles_1044 set mol = mol_from_smiles(smiles::cstring);

#create index
create index mol_index on dpph.avail_smiles using gist(mol);
create index mol_index_1044 on dpph.avail_smiles_1044 using gist(mol);