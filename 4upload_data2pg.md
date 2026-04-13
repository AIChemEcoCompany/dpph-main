# copy smiles data to postgresql 
create datatase nida
create schema dpph
create table dpph.avail_smiles(smiles TEXT,cansmiles TEXT, price float, source TEXT, mol mol, ac smallint, amw float);
create table dpph.avail_smiles_1044(smiles TEXT, ac smallint, amw float, mol mol, is_463 smallint);
\copy dpph.avail_smiles(smiles, cansmiles, price, source, mol, ac, amw) from '/PATH/data' CSV HEADER;
\copy dpph.avail_smiles(smiles,  mol, ac, amw) from '/PATH/data' CSV HEADER;