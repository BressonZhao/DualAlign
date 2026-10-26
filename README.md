# DualAlign

> **Paper:** _[PAPER TITLE — placeholder]_
> **Authors:** _[AUTHORS — placeholder]_
> **Venue / Year:** _[VENUE YEAR — placeholder]_

DualAlign is a PyTorch / PyTorch Geometric implementation of **unsupervised graph domain
adaptation for node classification**. Given a labeled *source* graph and an unlabeled
*target* graph, it learns node representations that are aligned at two levels:

1. **Feature-level alignment (FS module).** An MLP encoder maps raw node features of both
   graphs into a shared space. A domain discriminator behind a gradient-reversal layer
   (DANN-style) pushes the encoder to make source and target features indistinguishable,
   while a classifier keeps them discriminative for the source labels.
2. **Graph-level alignment (GNN module).** A stack of GCN layers propagates the FS features
   over each graph. A second domain discriminator aligns the resulting graph embeddings.
3. **Structural alignment via edge reweighting (optional).** Every `rw_freq` epochs after
   `start_epoch`, each GCN layer gets a new per-edge weight vector for the *source* graph.
   Source edges are binned by the embedding distance between their endpoints. Each bin's
   weight is the target edge density for that bin divided by the source edge density. The
   weights are blended into message passing as `(1 - rw_lmda) + rw_lmda * w_e`.

Training runs in **two stages**:

| Stage | Script | What it does | Output |
| --- | --- | --- | --- |
| 1 | `main/train_MLP.py` | Pretrains the FS module: MLP encoder, MLP domain discriminator and a feature-only classifier | `best_freeze_model<seed>.pth` |
| 2 | `main/main.py` | Loads the stage-1 FS weights for the same seed, then jointly trains FS + GCN layers + both discriminators + classifier, with optional edge reweighting | `best_free_*_<seed>.pt`, `best_edge_weights_<seed>.pt` |

The FS module is **not frozen** in stage 2. It is initialized from stage 1 and fine-tuned.
The file names (`best_freeze_model…`, `best_free_…`) are historical.

---

## Repository structure

```
DualAlign/
├── README.md
├── CHANGELOG_FIXES.md          # list of bug fixes applied for the open-source release
├── requirements.txt            # original environment (Python 3.9, PyTorch 1.12.1, PyG 2.2.0)
├── main/
│   ├── train_MLP.py            # Stage 1: FS pretraining (MLP encoder + domain adversary)
│   ├── main.py                 # Stage 2: dual alignment (FS + GCN) with optional edge reweighting
│   └── models.py               # GradReverse, GCN_reweight, GS_reweight, MLPs, Adv_DANN, Classifier, GNN wrappers
├── Utils/
│   ├── utils.py                # losses, metrics, Logger, str2bool, seed-report aggregation
│   ├── edge_rw_distance.py     # distance-bin based source-edge reweighting
│   └── pre_data/
│       ├── datasets.py         # SBM generator and DBLP/ACM, Airport, Arxiv loaders
│       └── pre_arxiv.py        # temporal ogbn-arxiv subgraph construction
└── dataset/                    # NOT shipped — create it yourself (see "Datasets")
```

Both entry scripts resolve paths **relative to their own file location**, not the current
working directory:

- the repository root is the parent of `main/`, and is added to `sys.path`;
- dataset roots are under `<repo>/dataset/`;
- a relative `--dir_name` is resolved against `<repo>/main/`.

This means you can launch them from anywhere, e.g. `python DualAlign/main/train_MLP.py ...`.

---

## Requirements and installation

The code was developed with **Python 3.9, PyTorch 1.12.1 (CUDA 11.3) and PyTorch Geometric
2.2.0**. `requirements.txt` records that environment. Some entries in it are conda package
names (`pytorch`, `cudatoolkit`) and cannot be installed with `pip install -r`, so install
PyTorch and PyG first and then the remaining packages.

```bash
conda create -n dualalign python=3.9 -y
conda activate dualalign

# 1) PyTorch (pick the build matching your CUDA driver; CPU-only also works)
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch

# 2) PyTorch Geometric + compiled extensions.
#    Use the wheel index matching your torch/CUDA build; see
#    https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-scatter==2.1.0 torch-sparse==0.6.16 torch-cluster==1.6.0 \
    -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch-geometric==2.2.0

# 3) Remaining Python dependencies actually imported by the code
pip install numpy==1.23.5 scipy==1.10.0 scikit-learn==1.2.0 networkx==3.0
pip install ogb==1.3.5        # only needed for the Arxiv dataset
pip install optuna==4.1.0     # only needed for --search (hyperparameter search)
```

Notes:

- **`torch-sparse` is required.** `main/models.py` imports it at module load time.
- **`ogb`** is imported lazily and is only needed for `-d Arxiv`.
- **`optuna`** is imported lazily and is only needed for `--search`.
- **`nni`** is optional. If it is installed, stage 1 reports its final averaged metrics to
  NNI. Otherwise this step is skipped.
- Several packages in `requirements.txt` (e.g. `dgl`, `uproot`, `awkward`, `h5py`) are not
  imported by the current code. They are kept there as a record of the original environment.
- A GPU is used automatically when `torch.cuda.is_available()`. Checkpoints are loaded with
  `map_location`, so CPU-only runs also work.

---

## Datasets

No dataset files are included in this repository. Create `DualAlign/dataset/` and lay it
out as follows:

```
dataset/
├── acm/raw/{acm_docs.txt, acm_edgelist.txt, acm_labels.txt}
├── dblp/raw/{dblp_docs.txt, dblp_edgelist.txt, dblp_labels.txt}
├── Airport/
│   ├── usa/raw/{usa_labels.txt, usa_edgelist.txt}
│   ├── brazil/raw/{brazil_labels.txt, brazil_edgelist.txt}
│   └── europe/raw/{europe_labels.txt, europe_edgelist.txt}
└── ogbn_arxiv/ ...             # downloaded automatically by ogb on first use
```

| `--dataset` | Source / target | Files expected | Notes |
| --- | --- | --- | --- |
| `SBM` | generated | none | Two 3-block contextual stochastic block models (`3 * --num_nodes` nodes each). Target features are rotated by 30°. Controlled by `--num_nodes --sigma --ps --qs --pt --qt`. Generated per seed, deterministically from the seed. |
| `dblp_acm` | `--src_name`, `--tgt_name` (default `acm` → `dblp`) | `<name>_docs.txt` (comma-separated features, one node per line), `<name>_edgelist.txt` (`u,v` per line), `<name>_labels.txt` (one integer label per line, 0-based) | Edges are used in the direction given in the file. |
| `Airport` | `--src_name`, `--tgt_name` (e.g. `usa`, `brazil`, `europe`) | `<name>_labels.txt`, `<name>_edgelist.txt` | Node features are one-hot degrees (`OneHotDegree(241)`). |
| `Arxiv` | fixed time windows | downloaded by `ogb` into `dataset/` | Source = papers in (1950, 2007], target = papers in (2014, 2016]. These windows are hard-coded and `--start_year/--end_year` are ignored. The O(E) Python loops in the loader are slow. |

**Splits** (per class, reshuffled for every seed):

- source graph: 60% train / 20% validation / 20% test;
- target graph: 20% validation / 80% test.

Only **source training labels** are used in the loss. Target labels are used for
evaluation, and target-validation labels are also used for model selection when
`--valid_data tgt` (the default). Use `--valid_data src` to select models on source
validation accuracy only.

---

## Quick start (SBM, no data needed)

Run both stages with the **same `--dataset` and the same `--dir_name`**. Stage 2 reads the
stage-1 checkpoints from that directory. An absolute path is recommended:

```bash
cd DualAlign/main

# Stage 1: FS pretraining -> <dir_name>/best_freeze_model{1,3,5,6,8}.pth
python train_MLP.py -d SBM --dir_name /abs/path/to/results/sbm

# Stage 2: dual alignment with edge reweighting (method name contains "rw")
python main.py -d SBM -m DANN_rw --dir_name /abs/path/to/results/sbm
```

If you omit `--dir_name`, both stages use `DualAlign/main/result/`.

Each script runs the fixed seed list `[1, 3, 5, 6, 8]`. It prints a per-seed report and
then the mean ± std over seeds (population std). All console output is also appended to
`<dir_name>/log.dat`.

If stage 2 cannot find `best_freeze_model<seed>.pth`, it stops with an error telling you
to run stage 1 first.

To run stage 2 **without** edge reweighting, use a method name without `rw`, e.g.
`-m DANN`. The `--reweight` flag is overwritten by this rule in `main()`.

### Other datasets

```bash
# DBLP/ACM (ACM -> DBLP)
python train_MLP.py -d dblp_acm --src_name acm --tgt_name dblp --dir_name /abs/results/acm2dblp
python main.py      -d dblp_acm --src_name acm --tgt_name dblp --dir_name /abs/results/acm2dblp

# Airport (USA -> Brazil)
python train_MLP.py -d Airport --src_name usa --tgt_name brazil --dir_name /abs/results/usa2brazil
python main.py      -d Airport --src_name usa --tgt_name brazil --dir_name /abs/results/usa2brazil

# Arxiv (time-shift, fixed windows)
python train_MLP.py -d Arxiv --dir_name /abs/results/arxiv
python main.py      -d Arxiv --dir_name /abs/results/arxiv
```

### Hyperparameter search (optional, requires `optuna`)

```bash
python train_MLP.py --search -d SBM --dir_name /abs/results/sbm_search   # stage-1 search
python main.py      --search -d SBM --dir_name /abs/results/sbm_search   # stage-2 search (needs stage-1 ckpts)
```

Each search runs 50 trials, and every trial runs all five seeds. The objective is the
**mean target-validation accuracy**. Trial logs are written to
`main/search_log/search_results_stage{1,2}.txt`. Trials overwrite the checkpoints in
`--dir_name`, so use a separate directory for searches.

---

## Checkpoints

All checkpoints are written to `--dir_name` whenever the validation score improves:

| File | Stage | Content |
| --- | --- | --- |
| `best_freeze_model<seed>.pth` | 1 | dict with `MLP_encoder_state_dict`, `MLP_advesarial_state_dict`, `classifier_state_dict` |
| `best_free_H_l_encoder_<seed>.pt` | 2 | MLP encoder `state_dict` |
| `best_free_GNN_layer_<layer>_seed_<seed>.pt` | 2 | `state_dict` of GCN layer `layer` (0 … K-1) |
| `best_free_H_l_classifier_<seed>.pt` | 2 | classifier `state_dict` |
| `best_free_MLP_adversarial_<seed>.pt` | 2 | FS domain discriminator `state_dict` |
| `best_free_GNN_adversarial_<seed>.pt` | 2 | graph domain discriminator `state_dict` |
| `best_edge_weights_<seed>.pt` | 2 | list of K source-edge weight tensors `[E_src]` from the same epoch |

At the end of each seed, both stages reload the best checkpoint and report its metrics
(`--best_model true`, the default). Stage 2 also restores the **matching**
`best_edge_weights`. With `--best_model false`, the last-epoch model and edge weights are
reported instead.

The "validation score" is `acc_tgt_valid` (default, `--valid_data tgt`) or `acc_src_valid`
(`--valid_data src`). Test accuracy is never used for selection.

---

## Key arguments

Boolean flags take an explicit value: `true/false`, `yes/no` or `1/0`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `-d, --dataset` | `dblp_acm` | `SBM`, `dblp_acm`, `Airport`, `Arxiv` |
| `-m, --method` | `DANN` (stage 1) / `DANN_rw` (stage 2) | in stage 2, a name containing `rw` enables edge reweighting |
| `--src_name / --tgt_name` | `acm` / `dblp` | source/target graph names for `dblp_acm` and `Airport` |
| `--dir_name` | `result` | result/checkpoint directory; absolute, or relative to `main/` |
| `--valid_data` | `tgt` | model-selection split: `tgt` = target validation, `src` = source validation |
| `--best_model` | `true` | report the validation-selected checkpoint (else the last epoch) |
| `--epochs` | `300` | training epochs (stage 2's preset forces 300) |
| `--lr`, `--weight_decay` | `0.007`, `0` | Adam settings |
| `--opt_decay_step`, `--opt_decay_rate` | `50`, `0.8` | StepLR schedule |
| `--mlp_layers`, `--mlp_embed_dim`, `--hidden_dim` | `1`, `128`, `128` | FS encoder |
| `--K`, `--gcn_dim`, `--conv_dim` | `2`, `128`, `16` | number of GCN layers and their widths |
| `--cls_dim`, `--class_layers`, `--dc_layers` | `16`, `2`, `2` | classifier / discriminator widths and depths |
| `--alphatimes_mlp`, `--alphamin_mlp` | `1.5`, `1` | FS GRL coefficient `min(alphatimes * (epoch+1)/epochs, alphamin)` (`alphamin` is an upper cap) |
| `--alphatimes_gnn`, `--alphamin_gnn` | `1.5`, `1` | the same schedule for the graph-level GRL |
| `--alphatimes`, `--alphamin` | `1.5`, `1` | GRL schedule used in stage 1 |
| `--start_epoch`, `--rw_freq` | `200`, `20` | when reweighting starts and how often weights are refreshed |
| `--rw_lmda` | `0.5` | blend between plain GCN messages (0) and fully reweighted messages (1) |
| `--num_bins` | `10` | number of distance bins used by the reweighting |
| `--num_nodes --sigma --ps --qs --pt --qt` | `1000 0.8 0.2 0.02 0.2 0.016` | SBM generator |
| `--gpu_ratio` | `None` | optional `torch.cuda.set_per_process_memory_fraction` |

### Presets that override the command line

`main()` in both scripts applies a fixed preset **after** parsing, so some CLI values are
ignored. Edit the `optimized_params` dict in `main()` to change them.

- **Stage 1 (`train_MLP.py`):** `lr=0.007, mlp_embed_dim=128, cls_dim=16, hidden_dim=32,
  mlp_layers=1, alphamin=1, alphatimes=1`.
- **Stage 2 (`main.py`):** `epochs=300, lr=0.007, K=2, opt_decay_step=50, opt_decay_rate=0.8,
  hidden_dim=128, gcn_dim=64, conv_dim=64, rw_lmda=0.9, alphamin_gnn=0.5, alphatimes_gnn=1.5,
  alphamin_mlp=0.1, alphatimes_mlp=0.1, mlp_layers=1`. For `rw` methods it also sets
  `rw_freq=5, start_epoch=200`. `mlp_embed_dim` and `cls_dim` are **not** overridden and
  must match stage 1, because the FS weights are loaded from the stage-1 checkpoint.

---

## Known limitations

These are design or scope limitations, not bugs addressed by the fixes in
`CHANGELOG_FIXES.md`:

- **O(N²) memory for reweighting.** `edge_rw_distance.calculate_reweight_return` builds dense
  `N × N` distance matrices for both graphs (`torch.cdist`). This can exhaust GPU memory on
  large graphs such as Arxiv. Reduce the graph size, run reweighting on CPU, or use a
  non-`rw` method.
- **Full-batch training.** Both stages process whole graphs every epoch. There is no
  mini-batching or neighbor sampling.
- **Fixed seeds and presets.** The seed list `[1, 3, 5, 6, 8]` and the `optimized_params`
  presets are hard-coded in `main()`. `--seed` is not used to select seeds.
- **Target labels in model selection.** The default `--valid_data tgt` selects checkpoints
  with target-validation labels (20% of target nodes). For a strictly unsupervised
  protocol, use `--valid_data src`.
- **Optuna is optional.** `--search` needs `optuna`. Searches overwrite checkpoints in
  `--dir_name`.
- **Unused wrappers.** `GNN`, `GNN_simple` and `GNN_reweight` in `models.py` are not used
  by the training scripts. When `K > 3` they still share one `prop_hidden` module across
  intermediate layers. (The stage-2 training path builds a distinct layer per depth.)
  `GS_reweight` ignores its `reducer` argument.
- **Pileup.** `datasets.pileup` needs a `prepare_pileup` module that is not shipped. It
  raises an `ImportError` explaining this.
- **Data not shipped.** You need to obtain the DBLP/ACM and Airport files yourself (see
  "Datasets").
- **No LICENSE file yet.** See "License" below.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{PLACEHOLDER_KEY,
  title     = {[PAPER TITLE]},
  author    = {[AUTHORS]},
  booktitle = {[VENUE]},
  year      = {[YEAR]}
}
```

## License

Released under the [MIT License](LICENSE).

Before publishing, open `LICENSE` and replace `[COPYRIGHT HOLDER]` with the real copyright owner (and adjust the year if needed). If you prefer Apache-2.0 or another license, swap the file contents and update this section.


## Acknowledgements

The `GCN_reweight` layer is adapted from PyTorch Geometric's `GCNConv`. The Arxiv loader
uses the `ogbn-arxiv` dataset from the Open Graph Benchmark.
