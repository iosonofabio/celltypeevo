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
import matplotlib.pyplot as plt
import seaborn as sns


def get_data_folder(compressed=True):
    if compressed:
        return pathlib.Path('/home/fabio/university/PI/projects/compressed_atlas/prototype2/cell_atlas_approximations_API/web/static/atlas_data/')
    else:
        return pathlib.Path('/home/fabio/university/PI/projects/compressed_atlas/prototype2/cell_atlas_approximations_compression/data/full_atlases/gene_expression/')


def get_species():
    fns = os.listdir(get_data_folder())
    return [fn.split('.')[0] for fn in fns]


def get_atlas_file(species):
    return get_data_folder() / f'{species}.h5'


def get_number_types_express(species):
    fn = get_atlas_file(organism)
    with h5py.File(fn) as h5:
        me = h5['gene_expression']
        features = me['features'].asstr()[:]
        nfea = len(features)

        counts = np.zeros(nfea, np.int64)
        ncelltypes = 0
        for tissue, group in me['by_tissue'].items():
            frac = group['celltype']['fraction'][:, :]
            nz = (frac > 0).sum(axis=0)
            counts += nz
            ncelltypes += frac.shape[0]

    counts = pd.Series(counts, index=features)
    return counts, ncelltypes


def get_gini_coefficient(species, cell_state=False):
    fn = get_atlas_file(organism)
    with h5py.File(fn) as h5:
        me = h5['gene_expression']
        features = me['features'].asstr()[:]
        nfea = len(features)

        ginis = []
        for tissue, group in me['by_tissue'].items():
            if cell_state:
                avg = group['celltype']['neighborhood']['average'][:, :]
            else:
                avg = group['celltype']['average'][:, :]

            # Sort and compute cumulatives
            for i, avgi in enumerate(avg):
                tmp = np.sort(avgi)
                # Avg are in cptt, so divide by 10,000
                tmp /= 10000

                # Cumulative (finishes at one or thereabout)
                tmp = tmp.cumsum()

                # Area under Lorenz is just the mean (do the math)
                auL = tmp.mean()

                # Gini is (triangle - Lorenz) / triangle. The triangle has area 0.5
                gini = (0.5 - auL) / 0.5
                ginis.append(gini)

                #import ipdb; ipdb.set_trace()

    ginis = np.array(ginis)
    return ginis


if __name__ == '__main__':

    species = get_species()

    if False:
        print('Get fraction expressing')
        res = {}
        ncelltypes = {}
        for organism in species:
            print(organism)
            resi, ncelltypesi = get_number_types_express(organism)
            res[organism] = resi
            ncelltypes[organism] = ncelltypesi

        frac_expressed = pd.Series({key: (val != 0).mean() for key, val in res.items()}, name='frac_expressed')
        ngenes = pd.Series({key: len(val) for key, val in res.items()}, name='n_genes')
        df = pd.concat([frac_expressed, ngenes], axis=1)

        if False:
            fig, ax = plt.subplots(figsize=(3.5, 3))
            ax.scatter(df['n_genes'], df['frac_expressed'], color='k')
            for organism, row in df.iterrows():
                if row['frac_expressed'] < 0.75:
                    x = row['n_genes']
                    y = row['frac_expressed']
                    ax.text(x, y - 0.05, organism, ha='center', va='top')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True)
            ax.set_xlabel('Number of genes')
            ax.set_ylabel('Fraction of genes\nexpressed in at least 1 cell type')
            fig.tight_layout()
            #fig.savefig('../figures/frac_genes_expressed.svg')
            #fig.savefig('../figures/frac_genes_expressed.pdf')


        ncelltypes = pd.Series(ncelltypes, name='ncelltypes')
        res_frac = {key: 1.0 * val / ncelltypes[key] for key, val in res.items()}

        if False:
            organism_order = pd.Series({key: val.mean() for key, val in res_frac.items()}).sort_values().index
            colors = sns.color_palette('plasma', n_colors=len(res_frac))
            fig, ax = plt.subplots(figsize=(7.2, 3.2))
            for i, organism in enumerate(organism_order):
                fracs = res_frac[organism]
                x = np.sort(fracs.values) * 100
                y = (1.0 - np.linspace(0, 1, len(x))) * 100
                ax.plot(x, y, label=organism, color=colors[i])
            ax.set_xlabel('% cell types expressing')
            ax.set_ylabel('% genes expressed in\n> x % of cell types')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, ncols=2, title='Species:')
            ax.grid(True)
            fig.tight_layout()
            fig.savefig('../figures/pct_genes_expressed_in_pct_cell_types.svg')
            fig.savefig('../figures/pct_genes_expressed_in_pct_cell_types.pdf')


    if False:
        print('Get Gini coefficients')
        ginid = {}
        for organism in species:
            print(organism)
            resi = get_gini_coefficient(organism)
            ginid[organism] = resi

        if True:
            organism_order = pd.Series({key: val.mean() for key, val in ginid.items()}).sort_values().index
            colors = sns.color_palette('plasma', n_colors=len(organism_order))
            fig, ax = plt.subplots(figsize=(7.2, 3.2))
            for i, organism in enumerate(organism_order):
                ginis = ginid[organism]
                x = np.sort(ginis)
                y = (1.0 - np.linspace(0, 1, len(x)))
                ax.plot(x, y, label=organism, color=colors[i])
            ax.set_xlabel('Gini coefficient on transcripts')
            ax.set_ylabel('Fraction of cell types \nwith Gini > x')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, ncols=2, title='Species:')
            ax.grid(True)
            fig.tight_layout()
            fig.savefig('../figures/Gini_coeff_transcripts.svg')
            fig.savefig('../figures/Gini_coeff_transcripts.pdf')

    if True:
        print('Get Gini coefficients, cell states')
        ginid = {}
        for organism in species:
            print(organism)
            resi = get_gini_coefficient(organism, cell_state=True)
            ginid[organism] = resi

        if True:
            organism_order = pd.Series({key: val.mean() for key, val in ginid.items()}).sort_values().index
            colors = sns.color_palette('plasma', n_colors=len(organism_order))
            fig, ax = plt.subplots(figsize=(7.2, 3.2))
            for i, organism in enumerate(organism_order):
                ginis = ginid[organism]
                x = np.sort(ginis)
                y = (1.0 - np.linspace(0, 1, len(x)))
                ax.plot(x, y, label=organism, color=colors[i])
            ax.set_xlabel('Gini coefficient on transcripts')
            ax.set_ylabel('Fraction of cell states \nwith Gini > x')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, ncols=2, title='Species:')
            ax.grid(True)
            fig.tight_layout()
            fig.savefig('../figures/Gini_coeff_transcripts_cell_states.svg')
            fig.savefig('../figures/Gini_coeff_transcripts_cell_states.pdf')

        plt.ion(); plt.show()

