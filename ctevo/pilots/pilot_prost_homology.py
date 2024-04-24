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
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import parasail
except ImportError:
    parasail = None


fn_embedding = '../data/prost/prost_embeddings.h5'


def get_data_folder(compressed=True):
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


# Straight from PROST
def fasta_iter(fastafile):
    import gzip
    from itertools import groupby
    with gzip.open(fastafile, 'rt') as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
        for header in faiter:
            header = next(header)[1:].strip()
            seq = "".join(s.strip() for s in next(faiter))
            yield header, seq


def get_sequence_peptide(species, gene):
    """Get a peptide sequence from the inputs for ESM/PROST."""
    fn = get_data_folder(compressed=False) / species / 'peptide_sequences_for_esm.fasta.gz'
    for name, seq in fasta_iter(fn):
        if name == gene:
            return seq


def pretty_print_ali(ali1, ali2, term_width=None):
    if term_width is None:
        term_width = os.get_terminal_size().columns - 1
    nrows = len(ali1) // term_width  + 1 - int(len(ali1) % term_width == 0)
    print(len(ali1), term_width, nrows)
    for i in range(nrows):
        l1 = ali1[term_width * i: term_width * (i + 1)]
        l2 = ali2[term_width * i: term_width * (i + 1)]
        print(l1)
        print(l2)
        print()


if __name__ == '__main__':

    if False:
        species = get_species()
        t0 = time.time()
        print('Get feature embeddings')
        fembs = []
        for organism in ['h_sapiens', 'm_musculus', 'd_melanogaster', 'c_elegans']:
            print(organism)
            res = get_feature_embedding(organism)
            fembs.append(res)
        femb = xr.concat(fembs, 'dim_0')
        t1 = time.time()
        dt = t1 - t0
        print(dt, 'seconds')

    if False:
        print('Test a few genes between known species')

        def find_homologs(femb, gene1, species1, species2, n=10):
            f1 = femb.loc[femb.species == species1]
            f2 = femb.loc[femb.species == species2]
            return (np.abs(f1.loc[gene1].values - f2)).sum(axis=1).to_series().nsmallest(n)

        def find_distance_pairwise(femb, species1, species2):
            from scipy.spatial.distance import cdist

            f1 = femb.loc[femb.species == species1]
            f2 = femb.loc[femb.species == species2]

            block_size = 3000
            nb1 = (f1.shape[0] // block_size + 1) - int(f1.shape[0] % block_size == 0)
            nb2 = (f2.shape[0] // block_size + 1) - int(f2.shape[0] % block_size == 0)
            res = np.zeros((f1.shape[0], f2.shape[0]), np.float32)
            for i1 in range(nb1):
                f1b = f1.values[i1 * block_size: (i1 + 1) * block_size]
                for i2 in range(nb2):
                    print(f'{i1+1}/{nb1}, {i2+1}/{nb2}')
                    f2b = f2.values[i2 * block_size: (i2 + 1) * block_size]
                    dis = cdist(f1b, f2b, metric='cityblock')
                    res[i1 * block_size: (i1 + 1) * block_size, i2 * block_size: (i2 + 1) * block_size] = dis
            res = xr.DataArray(res, coords={'dim_0': f1.coords['feature'].values, 'dim_1': f2.coords['feature'].values})
            return res


        def find_closest_pairs(dis, n=10, exclude_identical=True):
            mat = dis.values.copy()
            vmax = mat.max() + 1
            if exclude_identical:
                mat[mat == 0] = vmax
            res = []
            for i in range(n):
                i1, i2 = np.unravel_index(mat.ravel().argmin(), mat.shape)
                val = mat[i1, i2]
                mat[i1, i2] = vmax
                res.append((i1, i2, dis.coords['dim_0'].values[i1], dis.coords['dim_1'].values[i2], val))
            return res

        dis = find_distance_pairwise(femb, 'h_sapiens', 'm_musculus')
        pairs = find_closest_pairs(dis, n=100, exclude_identical=False)


    if False:
        print('Test parasail')
        seq1 = 'GNNPWLIV'
        seq2 = 'PWIV'
        ali = parasail.sg_trace_striped_sat(seq1, seq2, 3, 1, parasail.blosum62)
        print(ali.traceback.query)
        print(ali.traceback.ref)

    if False:
        print('Get example genes and compute pairwise alignments')
        seq1 = get_sequence_peptide('h_sapiens', 'CD19')
        seq2 = get_sequence_peptide('m_musculus', 'Cd19')
        ali = parasail.sg_trace_striped_sat(seq1, seq2, 3, 1, parasail.blosum62)
        print('Human vs mouse')
        pretty_print_ali(ali.traceback.query, ali.traceback.ref)

        sys.exit()

        print('Human vs human refseq')
        cd19hrefseq = """MPPPRLLFFLLFLTPMEVRPEEPLVVKVEEGDNAVLQCLKGTSD
                         GPTQQLTWSRESPLKPFLKLSLGLPGLGIHMRPLAIWLFIFNVSQQMGGFYLCQPGPP
                         SEKAWQPGWTVNVEGSGELFRWNVSDLGGLGCGLKNRSSEGPSSPSGKLMSPKLYVWA
                         KDRPEIWEGEPPCLPPRDSLNQSLSQDLTMAPGSTLWLSCGVPPDSVSRGPLSWTHVH
                         PKGPKSLLSLELKDDRPARDMWVMETGLLLPRATAQDAGKYYCHRGNLTMSFHLEITA
                         RPVLWHWLLRTGGWKVSAVTLAYLIFCLCSLVGILHLQRALVLRRKRKRMTDPTRRFF
                         KVTPPPGSGPQNQYGNVLSLPTPTSGLGRAQRWAAGLGGTAPSYGNPSSDVQADGALG
                         SRSPPGVGPEEEEGEGYEEPDSEEDSEFYENDSNLGQDQLSQDGSGYENPEDEPLGPE
                         DEDSFSNAESYENEDEELTQPVARTMDFLSPHGSAWDPSREATSLAGSQSYEDMRGIL
                         YAAPQLRSIRGQPGPNHEEDADSYENMDNPDGPDPAWGGGGRMGTWSTR""".replace(' ', '').replace('\n', '')
        ali = parasail.sg_trace_striped_sat(seq1, cd19hrefseq, 3, 1, parasail.blosum62)
        pretty_print_ali(ali.traceback.query, ali.traceback.ref)

        print('Mouse vs mouse refseq')
        cd19mrefseq = """"MPSPLPVSFLLFLTLVGGRPQKSLLVEVEEGGNVVLPCLPDSSP
                         VSSEKLAWYRGNQSTPFLELSPGSPGLGLHVGSLGILLVIVNVSDHMGGFYLCQKRPP
                         FKDIWQPAWTVNVEDSGEMFRWNASDVRDLDCDLRNRSSGSHRSTSGSQLYVWAKDHP
                         KVWGTKPVCAPRGSSLNQSLINQDLTVAPGSTLWLSCGVPPVPVAKGSISWTHVHPRR
                         PNVSLLSLSLGGEHPVREMWVWGSLLLLPQATALDEGTYYCLRGNLTIERHVKVIARS
                         AVWLWLLRTGGWIVPVVTLVYVIFCMVSLVAFLYCQRAFILRRKRKRMTDPARRFFKV
                         TPPSGNGTQNQYGNVLSLPTSTSGQAHAQRWAAGLGSVPGSYGNPRIQVQDTGAQSHE
                         TGLEEEGEAYEEPDSEEGSEFYENDSNLGQDQVSQDGSGYENPEDEPMGPEEEDSFSN
                         AESYENADEELAQPVGRMMDFLSPHGSAWDPSREASSLGSQSYEDMRGILYAAPQLHS
                         IQSGPSHEEDADSYENMDKSDDLEPAWEGEGHMGTWGTT""".replace(' ', '').replace('\n', '')
        ali = parasail.sg_trace_striped_sat(seq2, cd19mrefseq, 3, 1, parasail.blosum62)
        pretty_print_ali(ali.traceback.query, ali.traceback.ref)

        print('Human refseq vs mouse refseq')
        ali = parasail.sg_trace_striped_sat(cd19hrefseq, cd19mrefseq, 3, 1, parasail.blosum62)
        pretty_print_ali(ali.traceback.query, ali.traceback.ref)


        for name, seq in fasta_iter('/home/fabio/Desktop/mouse_chrom7_uniprot.fasta.gz'):
            if 'GN=Cd19' in name:
                print(name)
                print(seq)


        for name, seq in fasta_iter('/home/fabio/Desktop/mart_export_mouse_peptides.fasta.gz'):
            if 'Cd19' in name:
                print(name)
                print(seq)

        for name, seq in fasta_iter('/home/fabio/Desktop/mouse_allchroms_uniprot.fasta.gz'):
            if 'GN=Cd19' in name:
                print(name)
                print(seq)

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

        if False:
            print('Follow one gene along evolution')
            def find_homologs_single_gene(femb, gene, species):
                fgene = femb.loc[femb.species == species].loc[gene].values
                
                homologs = {}
                organisms = np.unique(femb.species)
                for organism in organisms:
                    f2 = femb.loc[femb.species == organism]
                    homo_organism = (np.abs(f2 - fgene)).sum(axis=1).to_series().nsmallest(1)
                    homologs[organism] = {
                            'gene': homo_organism.index[0][0],
                            'distance': homo_organism.values[0],
                    }
                return pd.DataFrame(homologs).T

            res = find_homologs_single_gene(femb, 'CD19', 'h_sapiens')
            res = find_homologs_single_gene(femb, 'ACTB', 'h_sapiens')
            res3 = find_homologs_single_gene(femb, 'TPM3', 'h_sapiens')
            res_frog = find_homologs_single_gene(femb, 'rbp4l.S', 'x_laevis')


            def one_to_all_distance(femb, gene, species1, species2):
                fgene = femb.loc[femb.species == species1].loc[gene].values
                f2 = femb.loc[femb.species == species2]
                homo_organism = (np.abs(f2 - fgene)).sum(axis=1).to_series()
                return homo_organism

            dis = one_to_all_distance(femb, 'rbp4l.S', 'x_laevis', 'm_musculus')
            dish = one_to_all_distance(femb, 'RBP4', 'h_sapiens', 'm_musculus')

        if False:
            # Try to cluster genes into "macrogenes" a la SATURN
            from sklearn.cluster import HDBSCAN

            nrows = [10, 30, 100, 300] #, 1000, 3000, 10000]
            times = []
            for nrow in nrows:
                X = femb.values[:nrow]
                t0 = time.time()
                hdb = HDBSCAN(
                    min_cluster_size=4,
                    metric='l1',
                )
                hdb.fit(X)
                t1 = time.time()
                dt = t1 - t0
                times.append(dt)
                print(nrow, 'sequences', dt, 'seconds')

            if plt is not None:
                fig, ax = plt.subplots()
                ax.plot(nrows, times)
                ax.set_xlabel('# sequences')
                ax.set_ylabel('runtime clustering [s]')
                fig.tight_layout()
                plt.ion(); plt.show()


        if True:
            # Try to cluster genes into "macrogenes" a la SATURN, using GPU-knn + leiden
            import time
            import torch
            from pykeops.torch import LazyTensor
            
            use_cuda = torch.cuda.is_available()

            nrow = 100

            def compute_knn(femb, nrow=None, K=3, return_index=False):
                if nrow is None:
                    nrow = len(femb)
                x = torch.tensor(femb.values[:nrow].astype('float32'), device='cuda' if use_cuda else 'cpu')

                # N.B.: K has very little impact on the running time
                start = time.time()  # Benchmark:
                # Dimension-expanded self
                X_i = LazyTensor(x[:, None, :])
                # Dimension-expanded self, along a different axis
                X_j = LazyTensor(x[None, :, :])
                # Compute matrix of L1 distances
                # Notice the sum is, of course, along the non-expanded axis
                D_ij = ((X_i - X_j).abs()).sum(-1)
                
                ind_knn = D_ij.argKmin(K, dim=1)  # Samples <-> Dataset, (N_test, K)
                ind_knn_cpu = ind_knn.cpu()
                
                if use_cuda:
                    torch.cuda.synchronize()
                end = time.time()
                if not return_index:
                    return (end - start)
                return ind_knn, (end - start)

            if False:
                # Benchmarking runtimes, around 10 seconds for 100k sequences with a log10/log10 slope of 1.83
                nrows = [10, 30, 100, 300, 600, 1000, 3000, 6000, 10000, 30000, 100000]#, len(femb)]
                times = []
                for nrow in nrows:
                    print(nrow, 'sequences')
                    dt = compute_knn(femb, nrow)
                    print(dt)
                    times.append(dt)

                import plotext as plt
                plt.scatter(np.log10(nrows), np.log10(times))
                plt.title('Runtime for neighbors (CUDA)')
                plt.xlabel('Number of sequences (log10)')
                plt.ylabel('Runtime (log10) [s]')
                plt.show()            

            if False:
                def one_to_all_distance(femb, gene, species1, species2):
                    fgene = femb.loc[femb.species == species1].loc[gene].values
                    f2 = femb.loc[femb.species == species2]
                    homo_organism = (np.abs(f2 - fgene)).sum(axis=1).to_series()
                    return homo_organism

                j = 1
                ind1 = ind_knn_cpu[i, 0]
                ind2 = ind_knn_cpu[i, j]
                gene1, gene2 = femb.feature.values[[ind1, ind2]]
                species1, species2 = femb.species.values[[ind1, ind2]]
                dis = one_to_all_distance(femb, gene1, species1, species2)

            if False:
                K = 10
                print(f'Computing top {K} neighbors for each sequence in vertebrates')
                # Vertebrates, around 95k sequences
                femb_vert = femb[femb.species.isin(['h_sapiens', 'm_musculus', 'm_murinus', 'd_rerio', 'x_laevis'])]
                ind_knn, dt = compute_knn(femb_vert, K=K, return_index=True)
                ind_knn_cpu = ind_knn.cpu()

                from scipy.spatial.distance import pdist, squareform
                def self_dist(fembi):
                    d = squareform(pdist(fembi.values))
                    return d


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

            fembi = femb.loc[femb.feature.isin(['Ms4a1', 'Cd19', 'Cd2']) & (femb.species == 'm_musculus')]
            res = compute_target_species_equivalents(femb, fembi)


