# Stage 2 initializes FS from stage-one weights and jointly fine-tunes it with GCN layers.
# Domain heads act on both MLP features and graph embeddings; class loss uses source labels.
# Optional structural alignment updates a separate source-edge vector for each GCN layer.
# Imports, dataset roots, and relative --dir_name values are resolved from this file's
# location, so the script can be launched from any working directory.
import copy
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
from Utils import utils, edge_rw_distance
from Utils.utils import str2bool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Parse experiment options; the active main() preset overrides several of these values.
def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='dblp_acm')
    parser.add_argument('-m', '--method', type=str, help='method used', default='DANN_rw')
    # The active training function builds GCN_reweight directly, irrespective of --backbone.
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='GCN')
    # The runners use the fixed list [1, 3, 5, 6, 8] rather than this argument.
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

    # main()/objective() set reweight from whether the method name contains 'rw'.
    # The active reweighting routine uses embedding distances, not pseudo-labels.
    parser.add_argument('--reweight', type=str2bool, help='if we want to reweight the source graph', default=False)
    parser.add_argument('--rw_freq', type=int, help='every number of epochs to compute the new weights based on psuedo-labels', default=20)
    parser.add_argument('--start_epoch', type=int, help='starting epoch for reweighting', default=200)
    # rw_lmda mixes plain and reweighted messages in every GCN_reweight layer: (1-lmda) + lmda*w_e.
    parser.add_argument("--rw_lmda", type=float, help='lambda to control the rw', default=0.5)
    parser.add_argument("--use_valid_label", type=str2bool, help='if we want to use target validation ground truth label in rw', default=True)
    parser.add_argument('--pseudo', type=str2bool, help='if use pseudo label for reweighting', default=False)
    parser.add_argument("--rw_average", type=str2bool, help='if we want to average the edge weights when reweighting for multiple graphs', default=False)
    parser.add_argument("--alphatimes", type=float, help='constant in front of the alpha for DANN', default=1.5)
    # The alphamin* parameters are upper caps in min(ramp, cap), despite their names.
    parser.add_argument("--alphamin", type=float, help='min of alpha for DANN', default=1)
    parser.add_argument("--alphatimes_mlp", type=float, help='constant in front of the alpha for DANN(for FS)', default=1.5)
    parser.add_argument("--alphamin_mlp", type=float, help='min of alpha for DANN(for FS)', default=1)
    parser.add_argument("--alphatimes_gnn", type=float, help='constant in front of the alpha for DANN(for FCSS)', default=1.5)
    parser.add_argument("--alphamin_gnn", type=float, help='min of alpha for DANN(for FCSS)', default=1)

    # Relative paths are resolved from the main/ directory; must match the stage-one value.
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
    # Source/target domains use explicit time windows rather than the parsed year flags.
    elif args.dataset == "Arxiv":
        source_dataset = datasets.prepare_Arxiv(DATA_ROOT, [1950, 2007])
        target_dataset = datasets.prepare_Arxiv(DATA_ROOT, [2014, 2016])
        print("arxiv")
        print(source_dataset.num_edges)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset!r} (expected SBM, dblp_acm, Airport or Arxiv)")
    return source_dataset, target_dataset


# Build K GCN_reweight layers: mlp_embed_dim -> gcn_dim -> ... -> gcn_dim -> conv_dim.
# Every intermediate layer is a distinct module with its own parameters.
def build_gnn_layers(args):
    GNNs = [models.GCN_reweight(args.mlp_embed_dim, args.gcn_dim).to(device)]
    for i in range(args.K - 2):
        GNNs.append(models.GCN_reweight(args.gcn_dim, args.gcn_dim).to(device))
    GNNs.append(models.GCN_reweight(args.gcn_dim, args.conv_dim).to(device))
    return GNNs


# Checkpoint filenames written on validation improvement and reloaded for the final report.
def best_ckpt_paths(path, seed, K):
    return {
        'encoder': os.path.join(path, 'best_free_H_l_encoder_' + str(seed) + '.pt'),
        'gnn': [os.path.join(path, 'best_free_GNN_layer_' + str(layer) + '_seed_' + str(seed) + '.pt')
                for layer in range(K)],
        'classifier': os.path.join(path, 'best_free_H_l_classifier_' + str(seed) + '.pt'),
        'mlp_adversarial': os.path.join(path, 'best_free_MLP_adversarial_' + str(seed) + '.pt'),
        'gnn_adversarial': os.path.join(path, 'best_free_GNN_adversarial_' + str(seed) + '.pt'),
        'edge_weights': os.path.join(path, 'best_edge_weights_' + str(seed) + '.pt'),
    }


# Train one graph pair with feature/domain alignment and optional source-edge reweighting.
# MLP parameters remain trainable after initialization; the filename does not imply freezing.
def train_gnn_with_saved_mlp(source_dataset, target_dataset, args, seed):
    path = resolve_result_dir(args.dir_name)
    sys.stdout = utils.Logger(path)

    if isinstance(source_dataset, list):
        input_dim = source_dataset[0].num_node_features
        output_dim = source_dataset[0].num_classes
    else:
        input_dim = source_dataset.num_node_features
        output_dim = source_dataset.num_classes

    # Initialize FS from the stage-one checkpoint of the same seed.
    # The checkpoint must contain both MLP_encoder_state_dict and MLP_advesarial_state_dict.
    fs_ckpt = os.path.join(path, 'best_freeze_model' + str(seed) + '.pth')
    if not os.path.isfile(fs_ckpt):
        raise FileNotFoundError(
            f"Stage-one checkpoint not found: {fs_ckpt}\n"
            f"Run stage 1 first (python train_MLP.py ...) with the same --dataset and --dir_name.")
    checkpoint = torch.load(fs_ckpt, map_location=device)
    MLP_encoder = models.MLPs(input_dim, args).to(device)
    MLP_advesarial = models.Adv_DANN(args.mlp_embed_dim, args.cls_dim, 1, args.num_layers, args.dc_layers, args).to(
        device)
    MLP_encoder.load_state_dict(checkpoint['MLP_encoder_state_dict'])
    MLP_advesarial.load_state_dict(checkpoint['MLP_advesarial_state_dict'])

    # This path chains GCN layers directly, without the ReLU/dropout used in model wrappers.
    GNNs = build_gnn_layers(args)
    GNN_adversarial = models.Adv_DANN(args.conv_dim, args.cls_dim, 1, args.num_layers, args.dc_layers,
                                          args).to(device)
    classifier = models.Classifier(args.conv_dim, args.cls_dim, output_dim, args).to(device)
    best_MLP_encoder = models.MLPs(input_dim, args).to(device)
    best_GNNs = build_gnn_layers(args)
    best_classifier = models.Classifier(args.conv_dim, args.cls_dim, output_dim, args).to(device)
    best_edge_weights = None
    ckpt = best_ckpt_paths(path, seed, args.K)
    GNN_params = []
    for layer in range(args.K):
        GNN_params.extend(list(GNNs[layer].parameters()))
    # Jointly optimize the pretrained FS modules, graph layers, graph domain head, and classifier.
    total_params = (
            list(MLP_encoder.parameters()) +
            list(MLP_advesarial.parameters()) +
            GNN_params +
            list(GNN_adversarial.parameters()) +
            list(classifier.parameters())
    )
    # This path uses Adam + StepLR directly, rather than --opt/--opt_scheduler dispatch.
    optimizer = optim.Adam(total_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    loss_src_train = []
    loss_src_valid = []
    loss_src_test = []
    loss_tgt_valid = []
    loss_tgt_test = []
    # Keep one [E_src] vector per layer; target edges retain their original weights.
    edge_weights = [source_dataset.edge_weight.to(device).clone() for _ in range(args.K)]
    # Start below any attainable score so the first evaluated epoch always produces a checkpoint.
    best_valid_score = -float('inf')
    best_epoch = 0
    rw_rec = 0
    for epoch in range(args.epochs):
        MLP_encoder.train()
        MLP_advesarial.train()
        for layer in range(args.K):
            GNNs[layer].train()
        GNN_adversarial.train()
        classifier.train()


        src_data = source_dataset.to(device)
        tgt_data = target_dataset.to(device)

        # FS maps both domains into the same mlp_embed_dim feature space.
        mlp_src = MLP_encoder(src_data)
        mlp_tgt = MLP_encoder(tgt_data)

        gnn_src = None
        gnn_tgt = None

        # Enter the reweighting branch once the zero-based epoch reaches start_epoch-1.
        # Each layer estimates its weights from its own current source/target embeddings.
        if args.reweight and epoch >= (args.start_epoch - 1):
            # Refresh only when (epoch-start_epoch-1) % rw_freq == 0; epoch is zero-based.
            refresh = (epoch - args.start_epoch - 1) % args.rw_freq == 0
            if refresh:
                rw_rec += 1
            for layer in range(args.K):
                if layer == 0:
                    embed_src = GNNs[layer](mlp_src, src_data.edge_index, edge_weights[layer], lmda=args.rw_lmda)
                    embed_tgt = GNNs[layer](mlp_tgt, tgt_data.edge_index, tgt_data.edge_weight, lmda=args.rw_lmda)
                else:
                    embed_src = GNNs[layer](gnn_src, src_data.edge_index, edge_weights[layer], lmda=args.rw_lmda)
                    embed_tgt = GNNs[layer](gnn_tgt, tgt_data.edge_index, tgt_data.edge_weight, lmda=args.rw_lmda)

                # Detached embeddings make histogram-based weight updates external to autograd.
                if refresh:
                    edge_weight = edge_rw_distance.calculate_reweight_return(src_data, tgt_data,
                                                                             embed_src.clone().detach(),
                                                                             embed_tgt.clone().detach(),
                                                                             num_bins=args.num_bins, device=device)
                    edge_weights[layer] = edge_weight
                    # Recompute this layer with its new source weights before feeding the next layer.
                    if layer == 0:
                        gnn_src = GNNs[layer](mlp_src, src_data.edge_index, edge_weights[layer], lmda=args.rw_lmda)
                        gnn_tgt = GNNs[layer](mlp_tgt, tgt_data.edge_index, tgt_data.edge_weight, lmda=args.rw_lmda)
                    else:
                        gnn_src = GNNs[layer](gnn_src, src_data.edge_index, edge_weights[layer], lmda=args.rw_lmda)
                        gnn_tgt = GNNs[layer](gnn_tgt, tgt_data.edge_index, tgt_data.edge_weight, lmda=args.rw_lmda)
                else:
                    gnn_src = embed_src
                    gnn_tgt = embed_tgt
        else:
            # Before reweighting starts, propagate with the stored vectors (initially all ones).
            for layer in range(args.K):
                if layer == 0:
                    gnn_src = GNNs[layer](mlp_src, src_data.edge_index, edge_weights[layer], lmda=args.rw_lmda)
                    gnn_tgt = GNNs[layer](mlp_tgt, tgt_data.edge_index, tgt_data.edge_weight, lmda=args.rw_lmda)
                else:
                    gnn_src = GNNs[layer](gnn_src, src_data.edge_index, edge_weights[layer], lmda=args.rw_lmda)
                    gnn_tgt = GNNs[layer](gnn_tgt, tgt_data.edge_index, tgt_data.edge_weight, lmda=args.rw_lmda)

        # Schedule the graph-level and feature-level reversed gradients separately.
        alpha_gnn = min((args.alphatimes_gnn * (epoch + 1) / args.epochs), args.alphamin_gnn)
        alpha_mlp = min((args.alphatimes_mlp * (epoch + 1) / args.epochs), args.alphamin_mlp)


        final_mlp_src, pred_mlp_domain_src = MLP_advesarial(mlp_src, alpha_mlp)
        final_mlp_tgt, pred_mlp_domain_tgt = MLP_advesarial(mlp_tgt, alpha_mlp)

        # Both graph-domain branches use the graph-level coefficient alpha_gnn.
        final_src, pred_domain_src = GNN_adversarial(gnn_src, alpha_gnn)
        final_tgt, pred_domain_tgt = GNN_adversarial(gnn_tgt, alpha_gnn)

        pred_src = classifier(gnn_src)
        pred_tgt = classifier(gnn_tgt)

        # Restrict class supervision to source training nodes; use domain masks for both heads.
        mask_src = src_data.source_training_mask
        label_src = src_data.y[mask_src]
        pred_src = pred_src[mask_src]
        pred_domain_src = pred_domain_src[src_data.source_mask]
        pred_domain_tgt = pred_domain_tgt[tgt_data.target_mask]
        pred_mlp_domain_src = pred_mlp_domain_src[src_data.source_mask]
        pred_mlp_domain_tgt = pred_mlp_domain_tgt[tgt_data.target_mask]

        # Construct domain labels: source domain as 0, target domain as 1
        domain_label_src = torch.zeros_like(pred_domain_src)
        domain_label_tgt = torch.ones_like(pred_domain_tgt)

        # Compute source domain classification loss (cross-entropy loss)
        cls_loss_src = utils.CE_loss(pred_src, label_src)
        # Compute graph-embedding domain losses with binary cross-entropy.
        domain_loss_src = utils.BCE_loss(pred_domain_src, domain_label_src)
        domain_loss_tgt = utils.BCE_loss(pred_domain_tgt, domain_label_tgt)

        # Compute feature-embedding (MLP/FS) domain losses with binary cross-entropy.
        mlp_domain_loss_src = utils.BCE_loss(pred_mlp_domain_src, domain_label_src)
        mlp_domain_loss_tgt = utils.BCE_loss(pred_mlp_domain_tgt, domain_label_tgt)

        # Compute total loss as the sum of source classification loss and adversarial domain loss
        # GRL supplies the adversarial sign; the source CE and four domain BCE terms are added.
        loss = cls_loss_src + 1 * (domain_loss_src + domain_loss_tgt) + 1 * (mlp_domain_loss_src + mlp_domain_loss_tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 1 == 0:
            inter_report, loss_dict = evaluate(source_dataset, target_dataset, MLP_encoder, GNNs, classifier,
                                               edge_weights, args, )

            loss_src_train.append(loss_dict['loss_src_train'])
            loss_src_valid.append(loss_dict['loss_src_valid'])
            loss_src_test.append(loss_dict['loss_src_test'])
            loss_tgt_valid.append(loss_dict['loss_tgt_valid'])
            loss_tgt_test.append(loss_dict['loss_tgt_test'])

            # Model selection uses validation accuracy only: source validation ('src') or
            # target validation ('tgt', default). Test accuracy is never used for selection.
            if args.valid_data == "src":
                valid_score = inter_report['acc_src_valid']
            else:
                valid_score = inter_report['acc_tgt_valid']
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                best_epoch = epoch
                # Keep an in-memory copy of the best modules and their matching edge weights...
                best_MLP_encoder.load_state_dict(copy.deepcopy(MLP_encoder.state_dict()))
                for layer in range(args.K):
                    best_GNNs[layer].load_state_dict(copy.deepcopy(GNNs[layer].state_dict()))
                best_classifier.load_state_dict(copy.deepcopy(classifier.state_dict()))
                best_edge_weights = [w.detach().clone() for w in edge_weights]
                # ...and persist them so the selected model can be reloaded later.
                torch.save(MLP_encoder.state_dict(), ckpt['encoder'])
                for layer in range(args.K):
                    torch.save(GNNs[layer].state_dict(), ckpt['gnn'][layer])
                torch.save(classifier.state_dict(), ckpt['classifier'])
                torch.save(MLP_advesarial.state_dict(), ckpt['mlp_adversarial'])
                torch.save(GNN_adversarial.state_dict(), ckpt['gnn_adversarial'])
                torch.save([w.cpu() for w in best_edge_weights], ckpt['edge_weights'])

    print("best_valid_score: " + str(best_valid_score))
    if args.best_model:
        # Reload the validation-selected checkpoint together with the edge weights saved at that epoch.
        best_MLP_encoder.load_state_dict(torch.load(ckpt['encoder'], map_location=device))
        for layer in range(args.K):
            best_GNNs[layer].load_state_dict(torch.load(ckpt['gnn'][layer], map_location=device))
        best_classifier.load_state_dict(torch.load(ckpt['classifier'], map_location=device))
        best_edge_weights = [w.to(device) for w in torch.load(ckpt['edge_weights'], map_location=device)]
        final_report, loss_dict = evaluate(source_dataset, target_dataset, best_MLP_encoder, best_GNNs,
                                           best_classifier, best_edge_weights, args)
    else:
        # Evaluate the last-epoch model with its last-epoch edge weights.
        final_report, loss_dict = evaluate(source_dataset, target_dataset, MLP_encoder, GNNs, classifier,
                                           edge_weights, args)
    print("rw times: " + str(rw_rec))
    print("best_epoch: " + str(best_epoch))
    print("final_acc_tgt_test: " + str(final_report['acc_tgt_test']))
    return final_report




# Collect split-wise cross-entropy, accuracy, and AUC without domain-head predictions.
def evaluate(source_dataset, target_dataset, MLP_encoder, GNN, classifier,edge_weights, args):

    loss_src_train, acc_src_train, auc_src_train = evaluate_dataset(
        source_dataset, MLP_encoder, GNN, classifier,edge_weights, args, "src_train"
    )
    loss_src_valid, acc_src_valid, auc_src_valid = evaluate_dataset(
        source_dataset, MLP_encoder, GNN, classifier, edge_weights, args,"src_valid"
    )
    loss_src_test, acc_src_test, auc_src_test = evaluate_dataset(
        source_dataset, MLP_encoder, GNN, classifier, edge_weights, args,"src_test"
    )
    loss_tgt_valid, acc_tgt_valid, auc_tgt_valid = evaluate_dataset(
        target_dataset, MLP_encoder, GNN, classifier, edge_weights, args,"tgt_valid"
    )
    loss_tgt_test, acc_tgt_test, auc_tgt_test = evaluate_dataset(
        target_dataset, MLP_encoder, GNN, classifier, edge_weights, args,"tgt_test"
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


# Run the full graph, then restrict predictions and labels to the requested node split.
# Source phases use the learned layer-wise weights; target phases use original edge weights.
def evaluate_dataset(data, MLP_encoder, GNNs, classifier, edge_weights, args, phase):
    """
    Evaluate on the specified dataset type.
    """
    data = data.to(device)
    MLP_encoder.eval()
    for layer in range(args.K):
        GNNs[layer].eval()
    classifier.eval()
    with torch.no_grad():
        # Determine the mask to use based on the phase.
        if phase == "src_train":
            mask = data.source_training_mask
        elif phase == "src_valid":
            mask = data.source_validation_mask
        elif phase == "src_test":
            mask = data.source_testing_mask
        elif phase == "tgt_valid":
            mask = data.target_validation_mask
        else:
            mask = data.target_testing_mask

        label = data.y
        mlp_features = MLP_encoder(data)

        if "src" in phase:
            current_edge_weight = edge_weights
        else:
            current_edge_weight = [data.edge_weight.to(device).clone() for _ in range(args.K)]

        for layer in range(args.K):
            if layer == 0:
                gnn_features = GNNs[layer](mlp_features, data.edge_index, current_edge_weight[layer],
                                           lmda=args.rw_lmda)
            else:
                gnn_features = GNNs[layer](gnn_features, data.edge_index, current_edge_weight[layer],
                                           lmda=args.rw_lmda)
        pred = classifier(gnn_features)

        pred = pred[mask]
        label = label[mask]

        cls_loss = utils.CE_loss(pred, label).item()

        acc, auc = utils.get_scores(pred, label)

    return cls_loss, acc, auc



# Optuna objective returning mean target validation accuracy across the fixed five seeds.
# Trial values below replace the corresponding parsed arguments.
def objective(trial):
    # Define the search space
    optimized_params = {
        'alphamin_gnn': trial.suggest_categorical('alphamin_gnn', [0.1,0.5,1]),
        'alphatimes_gnn': trial.suggest_categorical('alphatimes_gnn', [1.0,1.5,2]),
        'alphamin_mlp': trial.suggest_categorical('alphamin_mlp', [0.01,0.05,0.1,0.5]),
        'alphatimes_mlp': trial.suggest_categorical('alphatimes_mlp', [0.1, 0.5, 1.0, 1.5]),
        'gcn_dim': trial.suggest_categorical('gcn_dim', [32,64,128,256]),
        'conv_dim': trial.suggest_categorical('conv_dim', [16, 32, 64,128]),
        'start_epoch': trial.suggest_categorical('start_epoch', [200]),
        'rw_freq': trial.suggest_categorical('rw_freq', [1,5,10,15]),
        'rw_lmda': trial.suggest_categorical('rw_lmda', [0.5,0.6,0.7,0.8,0.9,1]),
    }

    #print("Optimized Parameters:", optimized_params)

    args = arg_parse()
    if args.gpu_ratio is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_ratio)

    args.reweight = False
    if 'rw' in args.method:
        args.reweight = True
        args.rw_freq = optimized_params['rw_freq']
        args.start_epoch = optimized_params['start_epoch']


    args.conv_dim = optimized_params['conv_dim']
    args.gcn_dim = optimized_params['gcn_dim']
    args.rw_lmda = optimized_params['rw_lmda']
    args.alphamin_mlp = optimized_params['alphamin_mlp']
    args.alphatimes_mlp = optimized_params['alphatimes_mlp']
    args.alphamin_gnn = optimized_params['alphamin_gnn']
    args.alphatimes_gnn = optimized_params['alphatimes_gnn']





    print("Arguments:", args)

    num_seeds = args.seed
    all_seeds = [1, 3, 5, 6, 8]
    reports = []
    best_scores = []
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
        report_dict = train_gnn_with_saved_mlp(source_dataset, target_dataset, args, seed)
        reports.append(report_dict)
        print('-' * 80), print('-' * 80), print(f'[Seed {seed} done]: ', json.dumps(report_dict, indent=4)), print(
            '-' * 80), print('-' * 80)

    avg_report, avg_std_report = utils.get_avg_std_report(reports)
    print(f'[All seeds done], Results: ', json.dumps(avg_std_report, indent=4))
    print('-' * 80), print('-' * 80), print('-' * 80), print('-' * 80)

    # Optimize mean target *validation* accuracy so that test nodes stay held out.
    return avg_report['acc_tgt_valid']

# Optional 50-trial Optuna search; run with `python main.py --search ...`.
# The objective is mean target validation accuracy over the fixed five seeds.
# Each trial needs the stage-one checkpoints best_freeze_model<seed>.pth in --dir_name.
def search(n_trials=50):
    try:
        import optuna
    except ImportError as e:
        raise ImportError("Hyperparameter search requires optuna: pip install optuna") from e

    # create an Optuna object
    study = optuna.create_study(direction='maximize')

    log_dir = os.path.join(SCRIPT_DIR, 'search_log')
    os.makedirs(log_dir, exist_ok=True)
    output_file = os.path.join(log_dir, 'search_results_stage2.txt')

    with open(output_file, "w") as f:
        f.write("Optuna Search Log\n")
        f.write("====================\n")

    # Define callback function used to record results
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

# Run the fixed dual-alignment preset; the script entry point calls this function.
def main():
    optimized_params = {
        'epochs': 300,
        'lr': 0.007,
        'K': 2,
        'alphamin_gnn': 0.5,
        'alphatimes_gnn': 1.5,
        'alphamin_mlp': 0.1,
        'alphatimes_mlp': 0.1,
        'opt_decay_step': 50,
        'opt_decay_rate': 0.8,
        'gcn_dim': 64,
        'hidden_dim': 128,
        'conv_dim': 64,
        'cls_dim': 16,
        'start_epoch': 200,
        'mlp_embed_dim': 128,
        'rw_freq': 5,
        'rw_lmda': 0.9,
        'mlp_layers': 1
    }
    # print(optimized_params)
    args = arg_parse()
    if args.gpu_ratio is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_ratio)

    args.reweight = False
    if 'rw' in args.method:
        print('args.reweight=True')
        args.reweight = True
        args.rw_freq = optimized_params['rw_freq']
        args.start_epoch = optimized_params['start_epoch']

    # These assignments override the corresponding command-line values.
    # The preset's mlp_embed_dim entry is not assigned here, so the parsed value remains.
    args.lr = optimized_params['lr']
    args.K = optimized_params['K']
    args.opt_decay_rate = optimized_params['opt_decay_rate']
    args.opt_decay_step = optimized_params['opt_decay_step']
    args.hidden_dim = optimized_params['hidden_dim']
    args.gcn_dim = optimized_params['gcn_dim']
    args.conv_dim = optimized_params['conv_dim']
    args.rw_lmda = optimized_params['rw_lmda']
    args.alphamin_gnn = optimized_params['alphamin_gnn']
    args.alphatimes_gnn = optimized_params['alphatimes_gnn']
    args.alphamin_mlp = optimized_params['alphamin_mlp']
    args.alphatimes_mlp = optimized_params['alphatimes_mlp']
    args.epochs = optimized_params['epochs']
    args.mlp_layers = optimized_params['mlp_layers']
    # args.dc_layers = optimized_params['dc_layers']
    # args.class_layers = optimized_params['class_layers']
    # args.dropout = optimized_params['dropout']

    print(args)
    num_seeds = args.seed
    all_seeds = [1, 3, 5, 6, 8]
    reports = []
    best_scores = []
    # Seed graph construction, split shuffling, and model initialization for each run.
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
        report_dict = train_gnn_with_saved_mlp(source_dataset, target_dataset, args, seed)
        reports.append(report_dict)
        print('-' * 80), print('-' * 80), print(f'[Seed {seed} done]: ', json.dumps(report_dict, indent=4)), print(
            '-' * 80), print('-' * 80)

    avg_report, avg_std_report = utils.get_avg_std_report(reports)
    print(f'[All seeds done], Results: ', json.dumps(avg_std_report, indent=4))
    print('-' * 80), print('-' * 80), print('-' * 80), print('-' * 80)


if __name__ == "__main__":
    # `--search` is consumed here (not by arg_parse) so the remaining flags still configure the trials.
    if '--search' in sys.argv:
        sys.argv.remove('--search')
        search()
    else:
        main()
