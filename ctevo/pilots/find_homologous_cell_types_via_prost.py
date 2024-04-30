# vim: fdm=indent
'''
author:     Fabio Zanini
date:       06/01/24
content:    Test cell type distance across species using embeddings.
'''
import time
import os
import sys
import pathlib
import h5py
import hdf5plugin
import anndata

import numpy as np
import pandas as pd
import xarray as xr
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


fn_embedding = '../data/prost/prost_embeddings.h5'


def get_data_folder(compressed=True):
    import platform
    if platform.node() == 'archacp1':
        return pathlib.Path('/home/fabio/projects/compressed_atlas/prototype2/cell_atlas_approximations_compression/data/atlas_approximations/')
    if compressed:
        return pathlib.Path('/home/fabio/university/PI/projects/compressed_atlas/prototype2/cell_atlas_approximations_API/web/static/atlas_data/')
    else:
        return pathlib.Path('/home/fabio/university/PI/projects/compressed_atlas/prototype2/cell_atlas_approximations_compression/data/full_atlases/gene_expression/')


def get_species():
    with h5py.File(fn_embedding) as h5_data:
        species = list(h5_data.keys())
    return species


def get_atlas_file(species):
    return get_data_folder() / f'{species}.h5'


def get_atlas(species, data_type='average'):
    fn = get_atlas_file(species)
    with h5py.File(fn) as h5:
        me = h5['measurements']['gene_expression']
        features = me['var_names'].asstr()[:]
        data = me['data']['tissue->celltype']
        obs = []
        X = []
        for tissue in data:
            celltypes = data[tissue]['obs_names'].asstr()[:]
            values = data[tissue][data_type][:, :]
            tissue_arr = np.array([tissue] * len(celltypes))
            obst = pd.DataFrame({
                'tissue': tissue_arr,
                'celltype': celltypes,
            })
            obst.index = obst['tissue'] + ':' + obst['celltype']
            obs.append(obst)
            X.append(values)
    obs = pd.concat(obs, axis=0)
    X = np.vstack(X)
    adata = anndata.AnnData(
        X=X,
        obs=obs,
    )
    adata.var_names = pd.Index(features, name='Genes')
    return adata


def get_feature_embedding(species):
    with h5py.File(fn_embedding) as h5_data:
        me = h5_data[species] 
        features = me['features'].asstr()[:]
        embeddings = me['embeddings'][:, :] / 128.

    mi = pd.MultiIndex.from_arrays([features, [species] * len(features)], names=['feature', 'species'])
    gemb = xr.DataArray(
        embeddings,
        coords={
            'dim_0': mi,
        },
    )
    idx = ~gemb.isnull().any(axis=1)    
    gemb = gemb[idx]
    return gemb


if __name__ == '__main__':

    if True:
        species = get_species()
        t0 = time.time()
        print('Get feature embeddings')
        fembs = []
        #for organism in ['h_sapiens', 'm_musculus', 'd_melanogaster', 'c_elegans']:
        for organism in species:
            print(organism)
            res = get_feature_embedding(organism)
            fembs.append(res)
        femb = xr.concat(fembs, 'dim_0')
        t1 = time.time()
        dt = t1 - t0
        print(dt, 'seconds')

        if True:
            # Try to cluster genes into "macrogenes" a la SATURN, using GPU-knn + leiden
            import time
            import torch
            from pykeops.torch import LazyTensor
            
            use_cuda = torch.cuda.is_available()

            def compute_target_species_equivalents(femb, fembi, target_species='h_sapiens', K=3):
                femb_target = femb[femb.species == target_species]
                x_query = torch.tensor(fembi.values.astype('float32'), device='cuda' if use_cuda else 'cpu')
                x_target = torch.tensor(femb_target.values.astype('float32'), device='cuda' if use_cuda else 'cpu')

                ind_res = - np.ones((len(fembi), K), int)
                X_j = LazyTensor(x_target, axis=1)
                for i in range(len(fembi)):
                    X_i = LazyTensor(x_query[i])
                    # Compute matrix of L1 distances
                    D_ij = ((X_i - X_j).abs()).sum(-1)
                    # Find nearest neighbor in target species 
                    ind_knn = D_ij.argKmin(K, dim=1).cpu()
                    ind_res[i] = np.asarray(ind_knn)
                
                    if use_cuda:
                        torch.cuda.synchronize()

                res = []
                for i in range(len(fembi)):
                    for j in range(K):
                        idx = ind_res[i, j]
                        tgt = femb_target[idx]
                        resi = {
                            'query_feature': fembi.feature.values[i],
                            'target_feature': tgt.feature.values,
                            'K': j + 1,
                            'distance': np.abs((fembi[i].values - tgt.values)).sum(axis=-1),
                        }
                        res.append(resi)
                res = pd.DataFrame(res)

                return res

            if False:
                fembi = femb.loc[femb.feature.isin(['Ms4a1', 'Cd19', 'Cd2']) & (femb.species == 'm_musculus')]
                res = compute_target_species_equivalents(femb, fembi)

            def get_markers(adata, tissue, celltype, n=10):
                adatat = adata[adata.obs['tissue'] == tissue]
                i = list(adatat.obs['celltype']).index(celltype)
                noni = [x for x in range(len(adatat)) if x != i]
                Xi = adatat.X[i]
                Xnoni = adatat.X[noni]

                # Take log??
                dX = pd.Series(
                    (-np.log(Xnoni + 1) + np.log(Xi + 1)).min(axis=0),
                    index=adatat.var_names,
                )
                markers = dX.nlargest(n)
                markers = markers.loc[markers > 0].index
                return markers

            def score_celltypes(adata, features):
                adataf = adata[:, list(features)]
                Xf = adataf.X.copy()
                Xf /= Xf.max(axis=0)
                Xf = np.log(Xf + 1)
                scores = Xf.sum(axis=1)
                adata.obs['cross-species score'] = scores
                order = np.argsort(scores)[::-1]
                return adataf[order]

            def find_sister_celltype(adata_query, adata_target, tissue_query, celltype_query):
                query = (adata_query.uns['species'], tissue_query, celltype_query)
                markers = get_markers(adata_query, query[1], query[2])
                fembi = femb.loc[femb.feature.isin(markers) & (femb.species == query[0])]
                markers_eq_table = compute_target_species_equivalents(
                    femb,
                    fembi,
                    target_species=adata_target.uns['species'],
                    K=1,
                )
                res = score_celltypes(
                    adata_target,
                    markers_eq_table['target_feature'].values,
                )
                row = res.obs.iloc[0]
                target = (adata_target.uns['species'], row['tissue'], row['celltype'])
                score = row['cross-species score']
                return (query, target, score)

            if False:
                print('Get target atlas')
                target_species = 'h_sapiens'
                adata_target = get_atlas(target_species)
                query = ('m_musculus', 'lung', 'fibroblast')
                adata_query = get_atlas(query[0])

                markers = get_markers(adata_query, query[1], query[2])
                fembi = femb.loc[femb.feature.isin(markers) & (femb.species == query[0])]
                markers_eq_table = compute_target_species_equivalents(femb, fembi, K=1)

                res = score_celltypes(
                    adata_target,
                    markers_eq_table['target_feature'].values,
                )

            if True:
                print('Bipartite graph between two species')
                species1 = 'h_sapiens'
                species2 = 'm_musculus'
                adata1 = get_atlas(species1)
                adata1.uns['species'] = species1
                adata2 = get_atlas(species2)
                adata2.uns['species'] = species2
                adatas = [adata1, adata2]
                bigraph = []
                for j in range(2):
                    adata_query = adatas[j]
                    adata_target = adatas[(j + 1) % 2]
                    for i, (_, row) in enumerate(adata_query.obs[['tissue', 'celltype']].iterrows()):
                        print(f'j: {j}, i: {i} / {len(adata_query)}')
                        query, target, score = find_sister_celltype(
                            adata_query, adata_target, row['tissue'], row['celltype'],
                        )
                        bigraph.append({
                            'source': query,
                            'target': target,
                            'score': score,
                        })
