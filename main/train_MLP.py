# To achieve better performance, the FS module (MLP + ADV) is pretrained and then fine-tuned with the GNN encoder.
# Stage 1 trains the feature-only encoder, a node classifier, and a domain head.
# Class supervision uses source training nodes; domain BCE uses both domains.
# Imports, dataset roots, and relative --dir_name values are resolved from this file's
# location, so the script can be launched from any working directory.
import json
import torch
import argparse
import numpy as np
import random
import os
import sys
import models
import torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_ROOT = os.path.join(REPO_ROOT, 'dataset')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from Utils.pre_data import datasets
from Utils import utils
from Utils.utils import str2bool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Parse the shared experiment options; several GNN/reweighting flags are unused here.
# main() and objective() override selected parsed values with their own settings.
def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    #utils.parse_optimizer(parser)
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='dblp_acm')
    parser.add_argument('-m', '--method', type=str, help='method used', default='DANN')
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='GCN')
    # The runners below use the fixed seeds [1, 3, 5, 6, 8], rather than this value.
    parser.add_argument('--seed', type=int, help='note this should be number of seeds', default=5)
    parser.add_argument('--gpu_ratio', type=float, help='gpu memory ratio', default=None)

    parser.add_argument('--batch_size', type=int, help='Training batch size', default=1)
    parser.add_argument('--num_layers', type=int, help='Number of embedding layers', default=0)
    parser.add_argument('--dc_layers', type=int, help='Number of domain classification layers', default=2)
    parser.add_argument('--mlp_layers', type=int, help='Number of MLP layers', default=1)
    parser.add_argument('--class_layers', type=int, help='Number of classification layers', default=2)
    parser.add_argument('--K', type=int, help='Number of GNN layers', default=2)
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension for GNN', default=128)
    parser.add_argument('--gcn_dim', type=int, help='hidden dimension for GNN', default=128)
    parser.add_argument('--mlp_embed_dim', type=int, help='output dimension for MLP', default=128)
    parser.add_argument('--conv_dim', type=int, help='output dimension for GNN layers', default=16)
    parser.add_argument('--cls_dim', type=int, help='hidden dimension for feature extractor layer', default=16)
    parser.add_argument('--bn', type=str2bool, help='if use batch normalization', default=False)
    parser.add_argument("--resnet", type=str2bool, help='if we want to use resnet', default=False)
    parser.add_argument('--best_model', type=str2bool,help='if use best model from validation results', default=True)
    parser.add_argument('--valid_data', type=str, help='src means use source valid to select model, tgt means use target valid', default='tgt')


    parser.add_argument('--epochs', type=int, help='Number of training epochs', default=300)
    parser.add_argument('--opt', type=str, help='optimizer', default='adam')
    parser.add_argument('--opt_scheduler', type=str, help='optimizer scheduler', default='step')
    parser.add_argument('--opt_decay_step', type=int, default=50)
    parser.add_argument('--opt_decay_rate', type=float, default=0.8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.007)

    parser.add_argument('--num_nodes', type=int, help='number of nodes for each block of Stochastic block model', default=1000)
    parser.add_argument("--ps", type=float, help="the source intraclass connection probability for SBM dataset", default=0.2)
    parser.add_argument("--qs", type=float, help="the source interclass connection probability for SBM dataset", default=0.02)
    parser.add_argument("--pt", type=float, help="the target intraclass connection probability for SBM dataset", default=0.2)
    parser.add_argument("--qt", type=float, help="the target interclass connection probability for SBM dataset", default=0.016)
    parser.add_argument('--sigma', type=float, help='sigma(std) in the stochastic block model', default=0.8)
    parser.add_argument('--theta', type=float, help='theta in the stochastic block model', default=0)
    parser.add_argument('--sbm_alpha', type=float, help='alpha in the stochastic block model', default=0)

    parser.add_argument('--num_events', type=int, help='number of events for Pileup dataset', default=100)
    parser.add_argument('--edge_feature', type=str2bool, help='if we want to use edge feature during convolution', default=False)

    parser.add_argument('--reweight', type=str2bool, help='if we want to reweight the source graph', default=False)
    parser.add_argument('--rw_freq', type=int, help='every number of epochs to compute the new weights based on psuedo-labels', default=20)
    parser.add_argument('--start_epoch', type=int, help='starting epoch for reweighting', default=200)
    parser.add_argument("--rw_lmda", type=float, help='lambda to control the rw', default=0.5)
    # Legacy reweighting options are retained, but train_mlp does not consume them.
    parser.add_argument("--use_valid_label", type=str2bool, help='if we want to use target validation ground truth label in rw', default=True)
    parser.add_argument('--pseudo', type=str2bool, help='if use pseudo label for reweighting', default=False)
    parser.add_argument("--rw_average", type=str2bool, help='if we want to average the edge weights when reweighting for multiple graphs', default=False)
    parser.add_argument("--alphatimes", type=float, help='constant in front of the alpha for DANN', default=1.5)
    parser.add_argument("--alphamin", type=float, help='min of alpha for DANN', default=1)
    parser.add_argument("--alphatimes_mlp", type=float, help='constant in front of the alpha for DANN(for FS)', default=1.5)
    parser.add_argument("--alphamin_mlp", type=float, help='min of alpha for DANN(for FS)', default=1)
    parser.add_argument("--alphatimes_gnn", type=float, help='constant in front of the alpha for DANN(for FCSS)', default=1.5)
    parser.add_argument("--alphamin_gnn", type=float, help='min of alpha for DANN(for FCSS)', default=1)

    # Relative paths are resolved from the main/ directory; use the same value for both stages.
    parser.add_argument('--dir_name', type=str, help='result/checkpoint directory (absolute, or relative to main/)', default='result')
    parser.add_argument("--src_name", type=str, help='specify for the source dataset name',default='acm')
    parser.add_argument("--tgt_name", type=str, help='specify for the target dataset name',default='dblp')
    parser.add_argument('--start_year', type=int, help='training year start for arxiv', default=2005)
    parser.add_argument('--end_year', type=int, help='training year end for arxiv', default=2007)
    parser.add_argument("--balanced", type=str2bool, help='if we keep the label balanced in pileup dataset', default=True)
    parser.add_argument('--plt', type=str, help='plot using tsne or pca', default='pca')
    parser.add_argument("--num_bins", type=int, help='num_bins of calculate_reweight_based_on_distance', default=10)

    return parser.parse_args()


# Resolve --dir_name: absolute paths are used as-is, relative ones are anchored at main/.
def resolve_result_dir(dir_name):
    path = dir_name if os.path.isabs(dir_name) else os.path.abspath(os.path.join(SCRIPT_DIR, dir_name))
    os.makedirs(path, exist_ok=True)
    return path


# Build the source/target graph pair selected by --dataset (dataset roots live under DATA_ROOT).
def load_datasets(args):
    if args.dataset == "SBM":
        source_dataset = datasets.get_synthetic_source_data(args.num_nodes, args.sigma, args.ps, args.qs)
        target_dataset = datasets.get_synthetic_target_data(args.num_nodes, args.sigma, args.pt, args.qt,
                                                            rotation_angle=30)
    elif args.dataset == "dblp_acm":
        source_dataset = datasets.prepare_dblp_acm(DATA_ROOT, args.src_name)
        target_dataset = datasets.prepare_dblp_acm(DATA_ROOT, args.tgt_name)
    elif args.dataset == "Airport":
        source_dataset = datasets.prepare_airport(os.path.join(DATA_ROOT, 'Airport'), args.src_name)
        target_dataset = datasets.prepare_airport(os.path.join(DATA_ROOT, 'Airport'), args.tgt_name)
    # These hard-coded time windows take precedence over --start_year/--end_year.
    elif args.dataset == "Arxiv":
        source_dataset = datasets.prepare_Arxiv(DATA_ROOT, [1950, 2007])
        target_dataset = datasets.prepare_Arxiv(DATA_ROOT, [2014, 2016])
        print("arxiv")
        print(source_dataset.num_edges)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset!r} (expected SBM, dblp_acm, Airport or Arxiv)")
    return source_dataset, target_dataset




# Pretrain FS for one source/target graph pair and return split-wise metrics.
# The loaders supply node-index arrays in the attributes named *_mask.
def train_mlp(source_dataset, target_dataset, args, seed):
    path = resolve_result_dir(args.dir_name)
    sys.stdout = utils.Logger(path)
    ckpt_path = os.path.join(path, 'best_freeze_model' + str(seed) + '.pth')

    if isinstance(source_dataset, list):
        input_dim = source_dataset[0].num_node_features
        output_dim = source_dataset[0].num_classes
    else:
        input_dim = source_dataset.num_node_features
        output_dim = source_dataset.num_classes


    # The source classifier preserves task information while the domain head aligns features.
    MLP_encoder = models.MLPs(input_dim, args).to(device)
    best_MLP_encoder = models.MLPs(input_dim, args).to(device)
    MLP_advesarial = models.Adv_DANN(args.mlp_embed_dim, args.cls_dim, 1, args.num_layers, args.dc_layers, args).to(device)
    classifier = models.Classifier(args.mlp_embed_dim,args.cls_dim,output_dim,args).to(device)
    best_classifier = models.Classifier(args.mlp_embed_dim, args.cls_dim, output_dim, args).to(device)
    total_params = list(MLP_encoder.parameters()) + list(classifier.parameters())+ list(MLP_advesarial.parameters())
    optimizer = optim.Adam(total_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)

    loss_src_train = []
    loss_src_valid = []
    loss_src_test = []
    loss_tgt_valid = []
    loss_tgt_test = []
    # Start below any attainable score so the first evaluated epoch always produces a checkpoint.
    best_valid_score = -float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        MLP_encoder.train()
        classifier.train()
        MLP_advesarial.train()
        src_data = source_dataset.to(device)
        tgt_data = target_dataset.to(device)

        # Linearly ramp the reversed gradient and cap it at alphamin (despite the name).
        alpha = min((args.alphatimes * (epoch + 1) / args.epochs), args.alphamin)

        mlp_src = MLP_encoder(src_data)
        mlp_tgt = MLP_encoder(tgt_data)

        final_src, pred_domain_src = MLP_advesarial(mlp_src, alpha)
        final_tgt, pred_domain_tgt = MLP_advesarial(mlp_tgt, alpha)

        pred_src = classifier(mlp_src)
        pred_tgt = classifier(mlp_tgt)

        # Use source class labels only on source_training_mask; target labels are not in this loss.
        # Domain supervision uses source_mask and target_mask, which may cover more nodes.
        mask_src = src_data.source_training_mask
        label_src = src_data.y[mask_src]
        pred_src = pred_src[mask_src]
        pred_domain_src = pred_domain_src[src_data.source_mask]
        pred_domain_tgt = pred_domain_tgt[tgt_data.target_mask]

        # Construct domain labels: source domain as 0, target domain as 1
        domain_label_src = torch.zeros_like(pred_domain_src)
        domain_label_tgt = torch.ones_like(pred_domain_tgt)

        # Compute source domain classification loss (cross-entropy loss)
        cls_loss_src = utils.CE_loss(pred_src, label_src)

        # Compute domain classification loss (using binary cross-entropy)
        domain_loss_src = utils.BCE_loss(pred_domain_src, domain_label_src)
        domain_loss_tgt = utils.BCE_loss(pred_domain_tgt, domain_label_tgt)

        # GRL already supplies the adversarial sign; all three losses are added positively.
        loss = cls_loss_src + 1 * (domain_loss_src+domain_loss_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


        # Evaluate every epoch, using labels only for reporting and checkpoint selection.
        if (epoch+1) % 1 == 0:
            inter_report, loss_dict = evaluate(source_dataset, target_dataset, MLP_encoder, classifier)

            loss_src_train.append(loss_dict['loss_src_train'])
            loss_src_valid.append(loss_dict['loss_src_valid'])
            loss_src_test.append(loss_dict['loss_src_test'])
            loss_tgt_valid.append(loss_dict['loss_tgt_valid'])
            loss_tgt_test.append(loss_dict['loss_tgt_test'])

            # The default selects checkpoints with target validation accuracy; 'src' uses source validation.
            if args.valid_data == "src":
                valid_score = inter_report['acc_src_valid']
            else:
                valid_score = inter_report['acc_tgt_valid']

            if valid_score > best_valid_score:
                best_epoch = epoch
                best_valid_score = valid_score
                # Save encoder, domain head, and classifier together for this seed.
                # Stage two loads best_freeze_model<seed>.pth for the matching seed.
                torch.save({
                    'MLP_encoder_state_dict': MLP_encoder.state_dict(),
                    'MLP_advesarial_state_dict': MLP_advesarial.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                }, ckpt_path)

    print("best_valid_score: " + str(best_valid_score))
    if args.best_model:
        # Reload the validation-selected checkpoint rather than evaluating the final epoch.
        checkpoint = torch.load(ckpt_path, map_location=device)
        best_MLP_encoder.load_state_dict(checkpoint['MLP_encoder_state_dict'])
        best_classifier.load_state_dict(checkpoint['classifier_state_dict'])
        final_report, loss_dict = evaluate(source_dataset, target_dataset, best_MLP_encoder, best_classifier)
    else:
        final_report, loss_dict = evaluate(source_dataset, target_dataset, MLP_encoder, classifier)

    print("final_acc_tgt_test: " + str(final_report['acc_tgt_test']))
    print("best_epoch: " + str(best_epoch))
    return final_report



# Collect loss, accuracy, and AUC for source train/validation/test and target phases.
# The report's 'default' key is target validation accuracy.
def evaluate(source_dataset, target_dataset, mlp_encoder,classifier):

    loss_src_train, acc_src_train, auc_src_train = evaluate_dataset(
        source_dataset, mlp_encoder,classifier, "src_train"
    )
    loss_src_valid, acc_src_valid, auc_src_valid = evaluate_dataset(
        source_dataset, mlp_encoder,classifier, "src_valid"
    )
    loss_src_test, acc_src_test, auc_src_test = evaluate_dataset(
        source_dataset, mlp_encoder,classifier, "src_test"
    )
    loss_tgt_valid, acc_tgt_valid, auc_tgt_valid = evaluate_dataset(
        target_dataset, mlp_encoder,classifier, "tgt_valid"
    )
    loss_tgt_test, acc_tgt_test, auc_tgt_test = evaluate_dataset(
        target_dataset, mlp_encoder,classifier, "tgt_test"
    )

    report = {
        "acc_tgt_valid": acc_tgt_valid,
        "auc_tgt_valid": auc_tgt_valid,
        "acc_tgt_test": acc_tgt_test,
        "auc_tgt_test": auc_tgt_test,
        "acc_src_valid": acc_src_valid,
        "auc_src_valid": auc_src_valid,
        "acc_src_train": acc_src_train,
        "auc_src_train": auc_src_train,
        "acc_src_test": acc_src_test,
        "auc_src_test": auc_src_test,
        "default": acc_tgt_valid,
    }
    loss_dict = {
        "loss_src_train": loss_src_train,
        "loss_src_valid": loss_src_valid,
        "loss_src_test": loss_src_test,
        "loss_tgt_valid": loss_tgt_valid,
        "loss_tgt_test": loss_tgt_test,
    }

    return report, loss_dict


# Compute feature-only predictions without gradients, then score the requested nodes.
def evaluate_dataset(data, mlp_encoder,classifier, phase):

    data = data.to(device)
    mlp_encoder.eval()
    classifier.eval()
    with torch.no_grad():
        if phase == "src_train":
            mask = data.source_training_mask
        elif phase == "src_valid":
            mask = data.source_validation_mask
        elif phase == "src_test":
            mask = data.source_testing_mask
        elif phase == "tgt_valid":
            mask = data.target_validation_mask
        else:
            # Held-out target test nodes only (disjoint from target_validation_mask), as in stage two.
            mask = data.target_testing_mask

        label = data.y
        mlp_features = mlp_encoder(data)
        pred = classifier(mlp_features)


        pred = pred[mask]
        label = label[mask]
        cls_loss = utils.CE_loss(pred, label).item()
        acc, auc = utils.get_scores(pred, label)


    return cls_loss, acc, auc


# Optuna objective: sample FS hyperparameters and return mean target validation accuracy.
# Each trial reruns all five seeds and can overwrite their checkpoint files.
def objective(trial):
    optimized_params = {
        'lr': trial.suggest_categorical('lr',  [0.007]),
        'mlp_embed_dim': trial.suggest_categorical('mlp_embed_dim', [64,128,512]),
        'cls_dim': trial.suggest_categorical('cls_dim', [32, 64,128]),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64,128,512]),
        'mlp_layers': trial.suggest_categorical('mlp_layers', [1, 2, 3]),
        'alphamin' : trial.suggest_categorical('alphamin', [0.1,0.5,1]),
        'alphatimes' : trial.suggest_categorical('alphatimes', [1.0,1.5,2]),
    }

    args = arg_parse()
    if args.gpu_ratio is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_ratio)

    args.reweight = False
    args.lr = optimized_params['lr']
    args.hidden_dim = optimized_params['hidden_dim']
    args.mlp_embed_dim = optimized_params['mlp_embed_dim']
    args.cls_dim = optimized_params['cls_dim']
    args.mlp_layers = optimized_params['mlp_layers']
    args.alphamin = optimized_params['alphamin']
    args.alphatimes = optimized_params['alphatimes']





    print("Arguments:", args)

    num_seeds = args.seed
    all_seeds = [1, 3, 5, 6, 8]
    best_scores = []
    reports = []
    for seed in all_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Each loader exposes the same graph and split attributes used by train_mlp.
        source_dataset, target_dataset = load_datasets(args)

        report_dict = train_mlp(source_dataset, target_dataset, args, seed)
        reports.append(report_dict)
        print('-' * 80), print('-' * 80), print(f'[Seed {seed} done]: ', json.dumps(report_dict, indent=4)), print(
            '-' * 80), print('-' * 80)

    avg_report, avg_std_report = utils.get_avg_std_report(reports)
    print(f'[All seeds done], Results: ', json.dumps(avg_std_report, indent=4))
    print('-' * 80), print('-' * 80), print('-' * 80), print('-' * 80)

    # Optimize mean target *validation* accuracy so that test nodes stay held out.
    return avg_report['acc_tgt_valid']

# Optional 50-trial Optuna search; run with `python train_MLP.py --search ...`.
# The objective is mean target validation accuracy over the fixed five seeds.
def search(n_trials=50):
    try:
        import optuna
    except ImportError as e:
        raise ImportError("Hyperparameter search requires optuna: pip install optuna") from e

    study = optuna.create_study(direction='maximize')

    log_dir = os.path.join(SCRIPT_DIR, 'search_log')
    os.makedirs(log_dir, exist_ok=True)
    output_file = os.path.join(log_dir, 'search_results_stage1.txt')

    
    with open(output_file, "w") as f:
        f.write("Optuna Search Log\n")
        f.write("====================\n")

    
    # Append each completed trial's objective and sampled parameters to the search log.
    def log_trial(study, trial):
        with open(output_file, "a") as f:  
            f.write(f"Trial {trial.number}:\n")
            f.write(f"  Value: {trial.value}\n")
            f.write("  Params:\n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")
            f.write("\n")

   
    study.optimize(objective, n_trials=n_trials, callbacks=[log_trial])

    
    with open(output_file, "a") as f:
        f.write("====================\n")
        f.write(f"Number of finished trials: {len(study.trials)}\n")
        f.write("Best trial:\n")

        trial = study.best_trial
        f.write(f"  Value: {trial.value}\n")
        f.write("  Params:\n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")

    print(f"Results have been saved to {output_file}")


# Run FS pretraining with the parameter preset below and the fixed five-seed list.
def main():

    optimized_params = {
            'lr': 0.007,
            'mlp_embed_dim': 128,
            'cls_dim': 16,
            'hidden_dim': 32,
            'mlp_layers': 1,
            'alphamin': 1,
            'alphatimes': 1
    }
    # print(optimized_params)
    args = arg_parse()
    if args.gpu_ratio is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_ratio)


    # These assignments override the corresponding command-line values.
    args.lr = optimized_params['lr']
    args.hidden_dim = optimized_params['hidden_dim']
    args.mlp_embed_dim = optimized_params['mlp_embed_dim']
    args.cls_dim = optimized_params['cls_dim']
    args.mlp_layers = optimized_params['mlp_layers']
    args.alphamin = optimized_params['alphamin']
    args.alphatimes = optimized_params['alphatimes']


    print(args)
    all_seeds = [1, 3, 5, 6, 8]
    reports = []
    for seed in all_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # prepare the dataset
        source_dataset, target_dataset = load_datasets(args)
        report_dict = train_mlp(source_dataset, target_dataset, args, seed)
        reports.append(report_dict)
        print('-' * 80), print('-' * 80), print(f'[Seed {seed} done]: ', json.dumps(report_dict, indent=4)), print(
            '-' * 80), print('-' * 80)

    avg_report, avg_std_report = utils.get_avg_std_report(reports)
    # Optional NNI reporting hook; skipped when nni is not installed.
    try:
        import nni
        nni.report_final_result(avg_report)
    except ImportError:
        pass
    print(f'[All seeds done], Results: ', json.dumps(avg_std_report, indent=4))
    print('-' * 80), print('-' * 80), print('-' * 80), print('-' * 80)



if __name__ == "__main__":
    # `--search` is consumed here (not by arg_parse) so the remaining flags still configure the trials.
    if '--search' in sys.argv:
        sys.argv.remove('--search')
        search()
    else:
        main()
