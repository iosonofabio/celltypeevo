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

import numpy as np
import pandas as pd
import xarray as xr


def get_data_folder():
    return pathlib.Path('/home/fabio/university/PI/projects/compressed_atlas/prototype2/cell_atlas_approximations_API/web/static/atlas_data/')


def get_species():
    fns = os.listdir(get_data_folder())
    return [fn.split('.')[0] for fn in fns]

def get_atlas_file(species):
    return get_data_folder() / f'{species}.h5'


def get_feature_embedding(species, features=None):
    with h5py.File(get_atlas_file(species)) as h5_data:
        me = h5_data['gene_expression']
        features = me['features'].asstr()[:]
        embeddings = me['esm2_embedding_layer33'][:, :]
    return xr.DataArray(
        embeddings,
        coords={'dim_0': features},
    )


def get_avg_expression(species):
    data = {'celltype': [], 'avg': [], 'organ': []}
    with h5py.File(get_atlas_file(species)) as h5_data:
        me = h5_data['gene_expression']
        for tissue in me['by_tissue']:
            gr = me['by_tissue'][tissue]['celltype']
            celltypesi = gr['index'].asstr()[:]
            if tissue == 'bladder':
                print(species, tissue, celltypesi)
            avgi = gr['fraction'][:, :]
            data['celltype'].append(celltypesi)
            data['organ'].append([tissue] * len(celltypesi))
            data['avg'].append(avgi)
    res = {
        'celltype': np.concatenate(data['celltype']),
        'organ': np.concatenate(data['organ']),
        'avg': np.vstack(data['avg']),
    }
    res['organism'] = [species] * len(res['organ'])
    mi = pd.MultiIndex.from_arrays([res['celltype'], res['organ'], res['organism']], names=['celltype', 'organ', 'organism'])
    res = pd.DataFrame(res['avg'].T, columns=mi)
    return res


def get_celltype_embedding(organism):
    print(organism)

    # Get gene embedding and average expression
    gemb = get_feature_embedding(organism)
    avg = get_avg_expression(organism)
    avg.index = gemb.coords['dim_0']    
    avg = xr.DataArray(avg)

    # Filter out NaNs
    idx = ~gemb.isnull().any(axis=1)    
    avg = avg[idx]
    gemb = gemb[idx]

    #FIXME
    # modify the avg with a better proxy
    # 1. blackout non-markers
    nmarkers = 10
    avg2 = avg.copy()
    for (ct, organ, _) in avg.coords['dim_1'].values:
        idx = (avg.coords['celltype'] != ct) & (avg.coords['organ'] == organ)
        diff = (avg.sel(celltype=ct) - avg.loc[:, idx].mean(axis=1)).isel(dim_1=0)
        avg2.loc[diff.sortby(diff, ascending=False)[nmarkers:].dim_0, ct] = 0
    avg = avg2

    # Cell embedding via matrix product
    mi = avg.coords['dim_1'].to_pandas().index
    mi.name = 'dim_0'
    cemb = xr.DataArray(
        avg.values.T @ gemb.values,
        coords={'dim_0': mi},
    )
    return cemb


if __name__ == '__main__':

    species = get_species()

    t0 = time.time()
    print('Get cell type embeddings')
    cembs = []
    for organism in species:
        res = get_celltype_embedding(organism)
        cembs.append(res)
    cemb = xr.concat(cembs, 'dim_0')
    t1 = time.time()
    dt = t1 - t0
    print(dt, 'seconds')

    print('UMAP the cell types')
    import umap
    from sklearn.preprocessing import StandardScaler    
    scaled_data = StandardScaler().fit_transform(cemb.values)    
    reducer = umap.UMAP()
    cumap = reducer.fit_transform(scaled_data)
    cumap = pd.DataFrame(cumap, index=cemb.coords['dim_0'].to_pandas().index, columns=['umap1', 'umap2']).reset_index()

    print('Plot')
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    gby = cumap.groupby('organism')
    colors = sns.color_palette('husl', n_colors=gby.ngroups)
    for i, (species, datum) in enumerate(gby):
        color = colors[i]
        if species in ('h_sapiens', 'm_murinus', 'd_rerio', 'x_laevis'):
            marker = 's'
        elif species in ('m_musculus',):
            marker = 'd'
        elif species in ('l_minuta',):
            marker = 'v'
        else:
            marker = 'o'
        
        ax.scatter(datum['umap1'], datum['umap2'], color=color, alpha=0.7, label=species, marker=marker)
    ax.legend(loc='upper left', bbox_transform=ax.transAxes, bbox_to_anchor=[1, 1], title='Organism:')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.ion(); plt.show()
    #fig.savefig('../figures/umap_cell_types_30_markers_esm2_18species.svg')
