# DualAlign (DACS)

> **Paper:** [Stage-Aware Dual Alignment for Covariate Shift in Graph Domain Adaptation](https://openreview.net/forum?id=nFn17qZZSS)  
> **Authors:** Hongwei Wen, Can Zhang, Haoyu He, Xintao Zhao, Hanyuan Hang, Minglong Lei  
> **Status:** Submitted to NeurIPS 2026 · [OpenReview](https://openreview.net/forum?id=nFn17qZZSS)

Unofficial PyTorch / PyG code for **DACS** (Dual Alignment for Covariate Shift): unsupervised graph domain adaptation for node classification under covariate shift in the joint feature–structure space.

Covariate shift is treated as two non-interchangeable stages of GNN computation:

1. **Feature Shift (FS)** — adversarial MLP alignment before message passing  
2. **Feature-Conditional Structure Shift (FCSS)** — layer-wise source-edge reweighting during propagation, plus a final graph-level adversarial alignment  

Training is **two-stage**:

| Stage | Script | Role |
| --- | --- | --- |
| 1 | `main/train_MLP.py` | Pretrain FS module → `best_freeze_model<seed>.pth` |
| 2 | `main/main.py` | Load stage-1 weights; train FS + GCN (+ optional edge RW) |

The FS module is **fine-tuned** in stage 2 (not frozen); the `freeze` / `free` filenames are historical.

See [`CHANGELOG_FIXES.md`](CHANGELOG_FIXES.md) for bug fixes applied in this release.

---

## Setup

Developed with **Python 3.9, PyTorch 1.12.1, PyG 2.2.0**. Install PyTorch / PyG for your CUDA build first, then:

```bash
pip install numpy scipy scikit-learn networkx
pip install ogb          # Arxiv only
pip install optuna       # --search only
```

`torch-sparse` is required at import time. Paths resolve relative to the scripts, so you can launch them from anywhere.

---

## Quick start (SBM, no data)

```bash
cd DualAlign/main
python train_MLP.py -d SBM --dir_name /abs/path/to/results/sbm
python main.py      -d SBM -m DANN_rw --dir_name /abs/path/to/results/sbm
```

Both stages share `--dir_name` and the fixed seed list `[1, 3, 5, 6, 8]`. Stage 2 needs the stage-1 checkpoints. Method names containing `rw` enable edge reweighting (`-m DANN` to disable).

Other datasets: `-d dblp_acm|Airport|Arxiv` with `--src_name` / `--tgt_name` as needed. Put raw files under `DualAlign/dataset/` (not shipped):

```
dataset/
├── acm/raw/   dblp/raw/
├── Airport/{usa,brazil,europe}/raw/
└── ogbn_arxiv/   # downloaded by ogb
```

Defaults: `--valid_data tgt` (target-val for selection), `--best_model true`. Use `--valid_data src` for a strictly unsupervised selection protocol.

---

## Layout

```
DualAlign/
├── main/          # train_MLP.py, main.py, models.py
├── Utils/         # losses, edge reweighting, dataset loaders
├── dataset/       # create locally (see above)
└── CHANGELOG_FIXES.md
```

Checkpoints land in `--dir_name`. Stage 2 also writes `best_edge_weights_<seed>.pt` when reweighting is on. Hyperparameter search: `--search` (needs Optuna; overwrites checkpoints in `--dir_name`).

**Caveats:** edge reweighting builds dense `N×N` distance matrices (memory-heavy on Arxiv); training is full-batch; `optimized_params` inside `main()` overrides some CLI values after parse.

---

## Citation

```bibtex
@inproceedings{wen2026dacs,
  title     = {Stage-Aware Dual Alignment for Covariate Shift in Graph Domain Adaptation},
  author    = {Wen, Hongwei and Zhang, Can and He, Haoyu and Zhao, Xintao and Hang, Hanyuan and Lei, Minglong},
  booktitle = {Submitted to NeurIPS},
  year      = {2026},
  url       = {https://openreview.net/forum?id=nFn17qZZSS}
}
```

## License

[MIT](LICENSE)
