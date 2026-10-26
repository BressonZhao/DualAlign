# Bug fixes for the open-source release

## `main/train_MLP.py` (stage 1)
- Removed the bare `nni.report_final_result(...)` call, which raised `NameError`. It is now
  wrapped in `try: import nni ... except ImportError: pass`.
- Target test metrics (`acc_tgt_test`, `auc_tgt_test`) now use `target_testing_mask`. They
  previously used `target_mask`, which also contains the validation nodes. This matches stage 2.
- Result directory: `--dir_name` is used as-is if absolute. Otherwise it is resolved against
  `main/`. The directory is created with `os.makedirs(..., exist_ok=True)`.
  The default changed from `../../DualAlign/main/result` to `result` (still `main/result`).
- Dataset roots are resolved from the repository root (`<repo>/dataset`), not the current
  working directory. The Airport root no longer starts with `/`. Dataset loading moved into
  one shared `load_datasets()` helper.
- The repository root is added to `sys.path` based on the script's location (was `sys.path.append('..')`).
- `best_valid_score` now starts at `-inf`, so a checkpoint is always written, even if
  validation accuracy is 0. `--best_model false` now reports the last-epoch model.
- Checkpoints are loaded with `map_location=device`, so CPU-only runs work.
- `search()` now creates `main/search_log/` and can be launched with `--search`. Optuna is
  imported lazily.
- The Optuna objective now returns mean target **validation** accuracy (previously target test accuracy).
- Removed the unused `edge_rw_distance` import and the module-level CUDA print.

## `main/main.py` (stage 2)
- The FS module is initialized from `best_freeze_model{seed}.pth` for the current seed
  (previously always `best_freeze_model5.pth`). If the file is missing, a clear
  `FileNotFoundError` says to run stage 1 first.
- On validation improvement, the best weights are copied in memory and saved to disk:
  `best_free_H_l_encoder_{seed}.pt`, `best_free_GNN_layer_{layer}_seed_{seed}.pt`,
  `best_free_H_l_classifier_{seed}.pt`, `best_free_MLP_adversarial_{seed}.pt`,
  `best_free_GNN_adversarial_{seed}.pt`, and `best_edge_weights_{seed}.pt`.
  Previously the script loaded these files at the end but never wrote them.
- The final evaluation reloads those checkpoints together with the **matching**
  `best_edge_weights`. Previously it used the last-epoch edge weights.
- `--valid_data`: `src` now selects on `acc_src_valid` and `tgt` on `acc_tgt_valid`.
  Previously both branches used `acc_tgt_test`.
- `lmda=args.rw_lmda` is passed to every `GCN_reweight` forward call in training and in
  evaluation. Previously the layer default `lmda=1` was always used.
- The target branch of the graph domain discriminator now uses `alpha_gnn`, like the source
  branch. It previously used `alpha_mlp`.
- For `K > 3`, every intermediate GCN layer is a distinct `GCN_reweight` module (in both
  `GNNs` and `best_GNNs`). Previously one `GNN_hidden` instance was shared.
- `search()` was a nested function that was never called. It is now a single function
  that runs Optuna, and can be launched with `--search`.
- Applied the same `--dir_name` / `DATA_ROOT` / `sys.path` handling as stage 1. The
  results directory was previously based on `../Utils/pre_data/`.
- `rw times` now counts the actual reweighting refreshes (it was always 0).
- The Optuna objective now returns mean target **validation** accuracy.

## Both scripts
- `type=bool` arguments (`--bn`, `--resnet`, `--best_model`, `--reweight`, `--edge_feature`,
  `--use_valid_label`, `--pseudo`, `--rw_average`, `--balanced`) now use
  `type=str2bool`. Previously `--flag False` was parsed as `True`.
- `--opt_decay_rate` changed from `type=int` to `type=float`. Previously `--opt_decay_rate 0.5` failed.
- Per-seed results are aggregated with the shared `utils.get_avg_std_report`. The two
  duplicated local copies were removed.
- An invalid `--dataset` now raises `ValueError`. Previously it continued with `None` datasets.

## `main/models.py`
- Removed the module-level `print(torch.cuda.is_available())`.
- `F.dropout(..., training=self.training)` in `GNN`, `GNN_simple`, `GNN_reweight` and
  `Adv_DANN`. Previously dropout stayed active after `.eval()`.
- `GCN_reweight`: the `# propagate_type:` comment now lists `lmda` and `edge_rw`. Newer
  PyG versions build `propagate()` from this comment and otherwise reject those kwargs.
  The comment has no effect on PyG 2.2.

## `Utils/utils.py`
- Added `str2bool`.
- Added `get_avg_std_report`. Missing values (`None`, e.g. an undefined AUC) become `NaN`
  and are skipped with `nanmean`/`nanstd`. Genuine `0.0` values are kept. Previously
  every falsy value was replaced by 0.
- `Logger`: wrapping an existing `Logger` reuses the original terminal and closes the
  previous log file. Previously each seed's log line was duplicated into all earlier
  loggers. `flush()` now flushes.

## `Utils/pre_data/datasets.py`
- `pileup()` raises a clear `ImportError` when `prepare_pileup` is not available.
  Previously it raised `NameError`.
- DBLP/ACM and Airport loaders: target index arrays (`target_validation_mask`,
  `target_testing_mask`, `target_mask`) are cast to `int`. They were float arrays, which
  cannot index tensors. The split logic is unchanged.
- `pre_arxiv` (and therefore `ogb`) is imported lazily inside `prepare_Arxiv`, so the
  other datasets do not require `ogb`.
