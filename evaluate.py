
import argparse
import copy
import json
import os
from itertools import product

import clip
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.stats import hmean
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import DATASET_PATHS
from models.compositional_modules import get_model
from utils import str2bool

cudnn.benchmark = True


DIR_PATH = os.path.dirname(os.path.realpath(__file__))




class Evaluator:
    """
    Evaluator class, adapted from:
    https://github.com/Tushar-N/attributes-as-operators

    With modifications from:
    https://github.com/ExplainableML/czsl
    """

    def __init__(self, dset, model):

        self.dset = dset

        # Convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe',
        # 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                 for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                            for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        # Mask over pairs that occur in closed world
        # Select set based on phase
        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            print('Evaluating with validation pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)

        self.test_pair_dict = [
            (dset.attr2idx[attr],
             dset.obj2idx[obj]) for attr,
            obj in test_pair_gt]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)

        # dict values are pair val, score, total
        for attr, obj in test_pair_gt:
            pair_val = dset.pair2idx[(attr, obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        # open world
        if dset.open_world:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        # masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        # Mask of seen concepts
        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        # Object specific mask over which pairs occur in the object oracle
        # setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # Decide if the model under evaluation is a manifold model or not
        self.score_model = self.score_manifold_model

    # Generate mask for each settings, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth, bias=0.0, topk=1):  # (Batch, #pairs)
        '''
        Inputs
            scores: Output scores
            obj_truth: Ground truth object
        Returns
            results: dict of results in 3 settings
        '''

        def get_pred_from_scores(_scores, topk):
            """
            Given list of scores, returns top 10 attr and obj predictions
            Check later
            """
            _, pair_pred = _scores.topk(
                topk, dim=1)  # sort returns indices of k largest values
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(
                -1, topk
            ), self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(
            scores.shape[0], 1
        )  # Repeat mask along pairs dimension
        scores[~mask] += bias  # Add bias to test pairs

        # Unbiased setting

        # Open world setting --no mask, all pairs of the dataset
        results.update({"open": get_pred_from_scores(scores, topk)})
        results.update(
            {"unbiased_open": get_pred_from_scores(orig_scores, topk)}
        )
        # Closed world setting - set the score for all Non test pairs to -1e10,
        # this excludes the pairs from set not in evaluation
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({"closed": get_pred_from_scores(closed_scores, topk)})
        results.update(
            {"unbiased_closed": get_pred_from_scores(closed_orig_scores, topk)}
        )

        return results

    def score_clf_model(self, scores, obj_truth, topk=1):
        '''
        Wrapper function to call generate_predictions for CLF models
        '''
        attr_pred, obj_pred = scores

        # Go to CPU
        attr_pred, obj_pred, obj_truth = attr_pred.to(
            'cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')

        # Gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # Multiply P(a) * P(o) to get P(pair)
        # Return only attributes that are in our pairs
        attr_subset = attr_pred.index_select(1, self.pairs[:, 0])
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset)  # (Batch, #pairs)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores

        return results

    def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=1):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        # Go to CPU
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.to(device)

        # Gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''

        results = {}
        # Repeat mask along pairs dimension
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias  # Add bias to test pairs

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10

        # sort returns indices of k largest values
        _, pair_pred = closed_scores.topk(topk, dim=1)
        # _, pair_pred = scores.topk(topk, dim=1)  # sort returns indices of k
        # largest values
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
            self.pairs[pair_pred][:, 1].view(-1, topk)

        results.update({'closed': (attr_pred, obj_pred)})
        return results

    def evaluate_predictions(
            self,
            predictions,
            attr_truth,
            obj_truth,
            pair_truth,
            allpred,
            topk=1):
        # Go to CPU
        attr_truth, obj_truth, pair_truth = (
            attr_truth.to("cpu"),
            obj_truth.to("cpu"),
            pair_truth.to("cpu"),
        )

        pairs = list(zip(list(attr_truth.numpy()), list(obj_truth.numpy())))

        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)

        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(
            unseen_ind
        )

        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            attr_match = (
                attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk]
            )
            obj_match = (
                obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk]
            )

            # Match of object pair
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            # Calculating class average accuracy

            seen_score, unseen_score = torch.ones(512, 5), torch.ones(512, 5)

            return attr_match, obj_match, match, seen_match, unseen_match, torch.Tensor(
                seen_score + unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score)

        def _add_to_dict(_scores, type_name, stats):
            base = [
                "_attr_match",
                "_obj_match",
                "_match",
                "_seen_match",
                "_unseen_match",
                "_ca",
                "_seen_ca",
                "_unseen_ca",
            ]
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        stats = dict()

        # Closed world
        closed_scores = _process(predictions["closed"])
        unbiased_closed = _process(predictions["unbiased_closed"])
        _add_to_dict(closed_scores, "closed", stats)
        _add_to_dict(unbiased_closed, "closed_ub", stats)

        # Calculating AUC
        scores = predictions["scores"]
        # getting score for each ground truth class
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][
            unseen_ind
        ]

        # Getting top predicted score for these unseen classes
        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[
            0][:, topk - 1]

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - correct_scores

        # Getting matched classes at max bias for diff
        unseen_matches = stats["closed_unseen_match"].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        # sorting these diffs
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats["closed_seen_match"].mean())
        unseen_match_max = float(stats["closed_unseen_match"].mean())
        seen_accuracy, unseen_accuracy = [], []

        # Go to CPU
        base_scores = {k: v.to("cpu") for k, v in allpred.items()}
        obj_truth = obj_truth.to("cpu")

        # Gather scores for all relevant (a,o) pairs
        base_scores = torch.stack(
            [allpred[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(
                scores, obj_truth, bias=bias, topk=topk)
            results = results['closed']  # we only need biased
            results = _process(results)
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(
            unseen_accuracy
        )
        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        try:
            harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis=0)
        except BaseException:
            harmonic_mean = 0

        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats["biasterm"] = float(bias_term)
        stats["best_unseen"] = np.max(unseen_accuracy)
        stats["best_seen"] = np.max(seen_accuracy)
        stats["AUC"] = area
        stats["hm_unseen"] = unseen_accuracy[idx]
        stats["hm_seen"] = seen_accuracy[idx]
        stats["best_hm"] = max_hm
        return stats


def compute_representations(model, test_dataset, config, device):
    """Function computes the attribute-object representations using
    the text encoder.
    Args:
        model (nn.Module): model
        test_dataset (CompositionDataset): CompositionDataset object
            with phase = 'test'
        config (argparse.ArgumentParser): config/args
        device (str): device type cpu/cuda:0
    Returns:
        torch.Tensor: returns the tensor with the attribute-object
            representations
    """
    obj2idx = test_dataset.obj2idx
    attr2idx = test_dataset.attr2idx
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                         for attr, obj in test_dataset.pairs]).to(device)

    attrs = torch.tensor([attr2idx[attr] for attr in test_dataset.attrs]).to(device)
    objs = torch.tensor([obj2idx[obj] for obj in test_dataset.objs]).to(device)

    test_pairs = np.array_split(
        pairs, len(pairs) // config.text_encoder_batch_size
    )

    num_attr_split = len(attrs) // config.text_encoder_batch_size
    num_obj_split = len(objs) // config.text_encoder_batch_size
    num_attr_split += 1 if num_attr_split == 0 else 0
    num_obj_split += 1 if num_obj_split == 0 else 0

    test_attrs = np.array_split(
        attrs, num_attr_split
    )
    test_objs = np.array_split(
        objs, num_obj_split
    )

    rep = torch.Tensor().to(device).type(model.dtype)
    rep_attr = torch.Tensor().to(device).type(model.dtype)
    rep_obj = torch.Tensor().to(device).type(model.dtype)
    with torch.no_grad():
        if 'hpl' in config.experiment_name:
            for batch_attr_obj in tqdm(test_pairs):
                batch_attr_obj = batch_attr_obj.to(device)
                token_tensors = model.construct_token_tensors_pair(batch_attr_obj)
                text_features = model.text_encoder(
                    model.token_ids,
                    token_tensors,
                    enable_pos_emb=model.enable_pos_emb,
                )
                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )
                rep = torch.cat([rep, text_features], dim=0)
            for batch_attr in tqdm(test_attrs):
                batch_attr = batch_attr.to(device)
                token_tensors_attr = model.construct_token_tensors_attr(batch_attr)
                text_features_attr = model.text_encoder(
                    model.token_ids,
                    token_tensors_attr,
                    enable_pos_emb=model.enable_pos_emb,
                )
                text_features_attr = text_features_attr / text_features_attr.norm(
                    dim=-1, keepdim=True
                )
                rep_attr = torch.cat([rep_attr, text_features_attr], dim=0)
            for batch_obj in tqdm(test_objs):
                batch_obj = batch_obj.to(device)
                token_tensors_obj = model.construct_token_tensors_obj(batch_obj)
                text_features_obj = model.text_encoder(
                    model.token_ids,
                    token_tensors_obj,
                    enable_pos_emb=model.enable_pos_emb,
                )
                text_features_obj = text_features_obj / text_features_obj.norm(
                    dim=-1, keepdim=True
                )
                rep_obj = torch.cat([rep_obj, text_features_obj], dim=0)
            return rep, rep_attr, rep_obj
        else:
            for batch_attr_obj in tqdm(test_pairs):
                batch_attr_obj = batch_attr_obj.to(device)
                token_tensors = model.construct_token_tensors(batch_attr_obj)
                text_features = model.text_encoder(
                    model.token_ids,
                    token_tensors,
                    enable_pos_emb=model.enable_pos_emb,
                )

                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )

                rep = torch.cat([rep, text_features], dim=0)
            return rep


def clip_baseline(model, test_dataset, config, device):
    """Function to get the clip representations.

    Args:
        model (nn.Module): the clip model
        test_dataset (CompositionDataset): the test/validation dataset
        config (argparse.ArgumentParser): config/args
        device (str): device type cpu/cuda:0

    Returns:
        torch.Tensor: returns the tensor with the attribute-object
            representations with clip model.
    """
    pairs = test_dataset.pairs
    pairs = [(attr.replace(".", " ").lower(),
              obj.replace(".", " ").lower())
             for attr, obj in pairs]

    prompts = [f"a photo of {attr} {obj}" for attr, obj in pairs]
    tokenized_prompts = clip.tokenize(
        prompts, context_length=config.context_length)
    test_batch_tokens = np.array_split(
        tokenized_prompts,
        len(tokenized_prompts) //
        config.text_encoder_batch_size)
    rep = torch.Tensor().to(device).type(model.dtype)
    with torch.no_grad():
        for batch_tokens in test_batch_tokens:
            batch_tokens = batch_tokens.to(device)
            _text_features = model.text_encoder(
                batch_tokens, enable_pos_emb=True)
            text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            rep = torch.cat((rep, text_features), dim=0)

    return rep


def predict_logits(model, text_rep, dataset, device, config, text_rep_attr=None, text_rep_obj=None):
    """Function to predict the cosine similarities between the
    images and the attribute-object representations. The function
    also returns the ground truth for attributes, objects, and pair
    of attribute-objects.

    Args:
        model (nn.Module): the model
        text_rep (nn.Tensor): the attribute-object representations.
        dataset (CompositionDataset): the composition dataset (validation/test)
        device (str): the device (either cpu/cuda:0)
        config (argparse.ArgumentParser): config/args

    Returns:
        tuple: the logits, attribute labels, object labels,
            pair attribute-object labels
    """
    model.eval()
    all_attr_gt, all_obj_gt, all_pair_gt = (
        [],
        [],
        [],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False)
    all_logits = torch.Tensor()
    if 'hpl' in config.experiment_name:
        all_logits_attr = torch.Tensor()
        all_logits_obj = torch.Tensor()
    with torch.no_grad():
        for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            batch_img = data[0].to(device)
            batch_img_feat = model.encode_image(batch_img)
            normalized_img = batch_img_feat / batch_img_feat.norm(
                dim=-1, keepdim=True
            )

            logits = (
                model.clip_model.logit_scale.exp()
                * normalized_img
                @ text_rep.t()
            )
            if 'hpl' in config.experiment_name:
                logits_attr = (
                        model.clip_model.logit_scale.exp()
                        * normalized_img
                        @ text_rep_attr.t()
                )
                logits_obj = (
                        model.clip_model.logit_scale.exp()
                        * normalized_img
                        @ text_rep_obj.t()
                )

            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
            logits = logits.cpu()
            all_logits = torch.cat([all_logits, logits], dim=0)

            if 'hpl' in config.experiment_name:
                logits_attr = logits_attr.cpu()
                logits_obj = logits_obj.cpu()
                all_logits_attr = torch.cat([all_logits_attr, logits_attr], dim=0)
                all_logits_obj = torch.cat([all_logits_obj, logits_obj], dim=0)

            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )
    if 'hpl' in config.experiment_name:
        return all_logits, all_attr_gt, all_obj_gt, all_pair_gt, all_logits_attr, all_logits_obj
    else:
        return all_logits, all_attr_gt, all_obj_gt, all_pair_gt


def threshold_with_feasibility(
        logits,
        seen_mask,
        threshold=None,
        feasiblity=None):
    """Function to remove infeasible compositions.

    Args:
        logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        seen_mask (torch.tensor): the seen mask with binary
        threshold (float, optional): the threshold value.
            Defaults to None.
        feasiblity (torch.Tensor, optional): the feasibility.
            Defaults to None.

    Returns:
        torch.Tensor: the logits after filtering out the
            infeasible compositions.
    """
    score = copy.deepcopy(logits)
    # Note: Pairs are already aligned here
    mask = (feasiblity >= threshold).float()
    # score = score*mask + (1.-mask)*(-1.)
    score = score * (mask + seen_mask)

    return score


def test(
        test_dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config,
        all_logits_attr=None,
        all_logits_obj=None,
):
    """Function computes accuracy on the validation and
    test dataset.

    Args:
        test_dataset (CompositionDataset): the validation/test
            dataset
        evaluator (Evaluator): the evaluator object
        all_logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        all_attr_gt (torch.tensor): the attribute ground truth
        all_obj_gt (torch.tensor): the object ground truth
        all_pair_gt (torch.tensor): the attribute-object pair ground
            truth
        config (argparse.ArgumentParser): the config

    Returns:
        dict: the result with all the metrics
    """
    # for itr, pair in enumerate(self.dset.pairs):
    #     scores[pair] = (1 - self.alpha) * score_pair[:, self.dset.all_pair2idx[pair]]
    #     scores[pair] += self.alpha * (score_attr[:, self.dset.attr2idx[pair[0]]] +
    #                                   score_obj[:, self.dset.obj2idx[pair[1]]])
    if 'hpl' in config.experiment_name and not config.open_world:
        if config.no_pair:
            predictions = {
                pair_name: config.alpha * (
                        all_logits_attr[:, test_dataset.attr2idx[pair_name[0]]] +
                        all_logits_obj[:, test_dataset.obj2idx[pair_name[1]]]
                )
                for i, pair_name in enumerate(test_dataset.pairs)
            }
        elif config.no_attr:
            predictions = {
                pair_name: (1 - config.alpha) * all_logits[:, i] + config.alpha * (
                    all_logits_obj[:, test_dataset.obj2idx[pair_name[1]]]
                )
                for i, pair_name in enumerate(test_dataset.pairs)
            }
        elif config.no_obj:
            predictions = {
                pair_name: (1 - config.alpha) * all_logits[:, i] + config.alpha * (
                        all_logits_attr[:, test_dataset.attr2idx[pair_name[0]]]
                )
                for i, pair_name in enumerate(test_dataset.pairs)
            }
        else:
            predictions = {
                pair_name: (1 - config.alpha) * all_logits[:, i] + config.alpha * (
                        all_logits_attr[:, test_dataset.attr2idx[pair_name[0]]] +
                        all_logits_obj[:, test_dataset.obj2idx[pair_name[1]]]
                )
                for i, pair_name in enumerate(test_dataset.pairs)
            }
        all_pred = [predictions]
    else:
        predictions = {
            pair_name: all_logits[:, i]
            for i, pair_name in enumerate(test_dataset.pairs)
        }
        all_pred = [predictions]

    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k] for i in range(len(all_pred))]
        ).float()

    results = evaluator.score_model(
        all_pred_dict, all_obj_gt, bias=config.bias, topk=config.topk
    )

    stats = evaluator.evaluate_predictions(
        results,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        all_pred_dict,
        topk=config.topk,
    )

    if config.topk == 1:
        attr_acc = float(torch.mean(
            (results['unbiased_closed'][0].squeeze(-1) == all_attr_gt).float()))
        obj_acc = float(torch.mean(
            (results['unbiased_closed'][1].squeeze(-1) == all_obj_gt).float()))
        stats['attr_acc'] = attr_acc
        stats['obj_acc'] = obj_acc

    return stats





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="name of the dataset", type=str)
    parser.add_argument(
        "--lr", help="learning rate", type=float, default=1e-04
    )
    parser.add_argument(
        "--weight_decay", help="weight decay", type=float, default=1e-05
    )
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-B/32"
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=64, type=int
    )
    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        type=str,
    )
    parser.add_argument(
        "--train_prefix",
        help="train with prefix a photo of",
        type=str2bool,
        default=False
    )
    parser.add_argument(
        "--evaluate_only",
        help="directly evaluate on the" "dataset without any training",
        action="store_true",
    )
    parser.add_argument(
        "--context_length",
        help="sets the context length of the clip model",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--attr_dropout",
        help="add dropout to attributes",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--open_world",
        help="evaluate on open world setup",
        action="store_true",
    )
    parser.add_argument(
        "--bias",
        help="eval bias",
        type=float,
        default=1e3,
    )
    parser.add_argument(
        "--topk",
        help="eval topk",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--soft_embeddings",
        help="location for softembeddings",
        type=str,
        default="./soft_embeddings.pt",
    )
    parser.add_argument(
        "--soft_embeddings_hpl",
        help="location for softembeddings_hpl",
        type=str,
        default="./soft_embeddings_hpl.pt",
    )
    parser.add_argument(
        "--soft_embeddings_prefix",
        help="",
        type=str,
        default="./soft_embeddings_prefix.pt",
    )

    parser.add_argument(
        "--text_encoder_batch_size",
        help="batch size of the text encoder",
        default=16,
        type=int,
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help="optional threshold"
    )
    parser.add_argument(
        '--threshold_trials',
        type=int,
        default=50,
        help="how many threshold values to try"
    )
    parser.add_argument(
        "--enable_img_transform",
        help="enable img transformation or not",
        default=0,
        type=int
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help="alpha * attr+obj + (1-alpha) * pair"
    )

    parser.add_argument(
        "--no_attr",
        help="ablation: w/o attr",
        action="store_true",
    )
    parser.add_argument(
        "--no_obj",
        help="ablation: w/o attr",
        action="store_true",
    )
    parser.add_argument(
        "--no_pair",
        help="ablation: w/o attr",
        action="store_true",
    )

    config = parser.parse_args()

    # set the seed value
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("evaluation details")
    print("----")
    print(f"dataset: {config.dataset}")
    print(f"experiment name: {config.experiment_name}")

    if config.experiment_name != 'clip':
        if not os.path.exists(config.soft_embeddings):
            print(f'{config.soft_embeddings} not found')
            print('code exiting!')
            exit(0)

    dataset_path = DATASET_PATHS[config.dataset]

    print('loading validation dataset')
    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural',
                                     open_world=config.open_world)

    print('loading test dataset')
    test_dataset = CompositionDataset(dataset_path,
                                      phase='test',
                                      split='compositional-split-natural',
                                      open_world=config.open_world)
    # get the model and the text rep
    if config.experiment_name == 'clip':
        clip_model, preprocess = load(
            config.clip_model, device=device, context_length=config.context_length)

        model = CLIPInterface(
            clip_model,
            config,
            token_ids=None,
            device=device,
            enable_pos_emb=True)
        val_text_rep = clip_baseline(model, val_dataset, config, device)
        test_text_rep = clip_baseline(model, test_dataset, config, device)
    else:
        model, optimizer = get_model(val_dataset, config, device)

        soft_embs = torch.load(config.soft_embeddings)['soft_embeddings']
        if 'hpl' in config.experiment_name:
            print('hpl')
            soft_embs_hpl = torch.load(config.soft_embeddings_hpl)['soft_embeddings_hpl']
            if config.train_prefix:
                print('train with prefix')
                soft_embs_prefix = torch.load(config.soft_embeddings_prefix)['soft_embeddings_prefix']
            else:
                print('train without prefix')
                soft_embs_prefix = None
            model.set_soft_embeddings(soft_embs, soft_embs_hpl,soft_embs_prefix)
            val_text_rep, val_text_rep_attr, val_text_rep_obj = compute_representations(
                model, val_dataset, config, device)
            test_text_rep, test_text_rep_attr, test_text_rep_obj = compute_representations(
                model, test_dataset, config, device)
        else:
            print('csp')
            if config.train_prefix:
                print('train with prefix')
                soft_embs_prefix = torch.load(config.soft_embeddings_prefix)['soft_embeddings_prefix']
            else:
                print('train without prefix')
                soft_embs_prefix = None
            model.set_soft_embeddings(soft_embs,soft_embs_prefix)
            val_text_rep = compute_representations(
                model, val_dataset, config, device)
            test_text_rep = compute_representations(
                model, test_dataset, config, device)

    print('evaluating on the validation set')
    skip_val = True

    if config.open_world and (config.threshold is None):
        evaluator = Evaluator(val_dataset, model=None)
        feasibility_path = os.path.join(
            DIR_PATH, f'data/feasibility_{config.dataset}.pt')
        unseen_scores = torch.load(
            feasibility_path,
            map_location='cpu')['feasibility']
        seen_mask = val_dataset.seen_mask.to('cpu')
        min_feasibility = (unseen_scores + seen_mask * 10.).min()
        max_feasibility = (unseen_scores - seen_mask * 10.).max()
        thresholds = np.linspace(
            min_feasibility,
            max_feasibility,
            num=config.threshold_trials)
        best_auc = 0.
        best_th = -10
        val_stats = None
        with torch.no_grad():
            if 'hpl' in config.experiment_name:
                all_logits, all_attr_gt, all_obj_gt, all_pair_gt, all_logits_attr, all_logits_obj = predict_logits(
                    model, val_text_rep, val_dataset, device, config, val_text_rep_attr, val_text_rep_obj)
                for i, pair_name in enumerate(test_dataset.pairs):
                    all_logits[:, i] = (1 - config.alpha) * all_logits[:, i] + config.alpha * (
                            all_logits_attr[:, test_dataset.attr2idx[pair_name[0]]] +
                            all_logits_obj[:, test_dataset.obj2idx[pair_name[1]]]
                    )
            else:
                all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits(
                    model, val_text_rep, val_dataset, device, config)
            for th in thresholds:
                temp_logits = threshold_with_feasibility(
                    all_logits, val_dataset.seen_mask, threshold=th, feasiblity=unseen_scores)
                results = test(
                    val_dataset,
                    evaluator,
                    temp_logits,
                    all_attr_gt,
                    all_obj_gt,
                    all_pair_gt,
                    config
                )
                auc = results['AUC']
                if auc > best_auc:
                    best_auc = auc
                    best_th = th
                    print('New best AUC', best_auc)
                    print('Threshold', best_th)
                    val_stats = copy.deepcopy(results)
    else:  # when best_th is given in args
        evaluator = Evaluator(val_dataset, model=None)
        with torch.no_grad():
            if 'hpl' in config.experiment_name:
                all_logits, all_attr_gt, all_obj_gt, all_pair_gt, all_logits_attr, all_logits_obj = predict_logits(
                    model, val_text_rep, val_dataset, device, config, val_text_rep_attr, val_text_rep_obj)
            else:
                all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits(
                    model, val_text_rep, val_dataset, device, config)
            if config.open_world:
                if 'hpl' in config.experiment_name:
                    for i, pair_name in enumerate(test_dataset.pairs):
                        all_logits[:, i] = (1 - config.alpha) * all_logits[:, i] + config.alpha * (
                                all_logits_attr[:, test_dataset.attr2idx[pair_name[0]]] +
                                all_logits_obj[:, test_dataset.obj2idx[pair_name[1]]]
                        )
                best_th = config.threshold
                feasibility_path = os.path.join(
                    DIR_PATH, f'data/feasibility_{config.dataset}.pt')
                unseen_scores = torch.load(
                    feasibility_path,
                    map_location='cpu')['feasibility']
                print('using threshold: ', best_th)
                all_logits = threshold_with_feasibility(
                    all_logits, val_dataset.seen_mask, threshold=best_th, feasiblity=unseen_scores)
            if not skip_val:
                if 'hpl' in config.experiment_name and not config.open_world:
                    results = test(
                        val_dataset,
                        evaluator,
                        all_logits,
                        all_attr_gt,
                        all_obj_gt,
                        all_pair_gt,
                        config,
                        all_logits_attr,
                        all_logits_obj
                    )
                else:
                    results = test(
                        val_dataset,
                        evaluator,
                        all_logits,
                        all_attr_gt,
                        all_obj_gt,
                        all_pair_gt,
                        config,
                    )
        if not skip_val:
            val_stats = copy.deepcopy(results)
            result = ""
            for key in val_stats:
                result = result + key + "  " + str(round(val_stats[key], 4)) + "| "
            print(result)

    print('evaluating on the test set')
    with torch.no_grad():
        evaluator = Evaluator(test_dataset, model=None)
        if 'hpl' in config.experiment_name:
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt, all_logits_attr, all_logits_obj = predict_logits(
                model, test_text_rep, test_dataset, device, config, test_text_rep_attr, test_text_rep_obj)
        else:
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits(
                model, test_text_rep, test_dataset, device, config)
        if config.open_world and best_th is not None:
            if 'hpl' in config.experiment_name:
                for i, pair_name in enumerate(test_dataset.pairs):
                    all_logits[:, i] = (1 - config.alpha) * all_logits[:, i] + config.alpha * (
                            all_logits_attr[:, test_dataset.attr2idx[pair_name[0]]] +
                            all_logits_obj[:, test_dataset.obj2idx[pair_name[1]]]
                    )
            print('using threshold: ', best_th)
            all_logits = threshold_with_feasibility(
                all_logits,
                test_dataset.seen_mask,
                threshold=best_th,
                feasiblity=unseen_scores)
        if 'hpl' in config.experiment_name and not config.open_world:
            test_stats = test(
                test_dataset,
                evaluator,
                all_logits,
                all_attr_gt,
                all_obj_gt,
                all_pair_gt,
                config,
                all_logits_attr,
                all_logits_obj
            )
        else:
            test_stats = test(
                test_dataset,
                evaluator,
                all_logits,
                all_attr_gt,
                all_obj_gt,
                all_pair_gt,
                config
            )

        result = ""
        for key in test_stats:
            result = result + key + "  " + \
                str(round(test_stats[key], 4)) + "| "
        print(result)

    if not skip_val:
        results = {
            'val': val_stats,
            'test': test_stats,
        }
    else:
        results = {
            'test': test_stats,
        }

    if config.open_world:
        if best_th is not None:
            results['best_threshold'] = best_th

    if config.experiment_name != 'clip':
        if config.open_world:
            result_path = config.soft_embeddings[:-2] + "open.calibrated.json"
        else:
            result_path = config.soft_embeddings[:-2] + "closed.json"

        with open(result_path, 'w+') as fp:
            json.dump(results, fp)

    print("done!")
