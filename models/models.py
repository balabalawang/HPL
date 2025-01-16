import os
import argparse

import clip
import pandas as pd
import torch
import torch.nn as nn
from clip_modules.interface import CLIPInterfaceHie, CLIPInterface
from clip_modules.model_loader import load

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class CSPInterface(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_embeddings,
        soft_embeddings_prefix: torch.nn.Parameter or None,
        class_token_ids,
        device="cuda:0",
        enable_pos_emb=True,
        attr_dropout=0.0,
    ):
        super().__init__(
            clip_model,
            config,
            class_token_ids,
            soft_embeddings,
            soft_embeddings_prefix,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )

        self.offset = offset
        self.attr_dropout = nn.Dropout(attr_dropout)

    def construct_token_tensors(self, pair_idx):
        """Function creates the token tensor for further inference.

        Args:
            pair_idx (torch.Tensor): Shape [N x 2], where N is the number
                of pairs of attr and obj

        Returns:
            torch.Tensor: token tensor passed to the text encoder;
                shape [N x context_length x 512]
        """
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype) 

        eos_idx = int(self.token_ids[0].argmax()) 
        soft_embeddings = self.attr_dropout(self.soft_embeddings)
        token_tensor[:, eos_idx - 2, :] = soft_embeddings[
            attr_idx
        ].type(self.clip_model.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_embeddings[
            obj_idx + self.offset
        ].type(self.clip_model.dtype)

        if self.config.train_prefix:
            token_tensor[:, 1 : len(self.soft_embeddings_prefix) + 1, :] = self.soft_embeddings_prefix.type(self.clip_model.dtype)

        return token_tensor

# init clip model
def csp_init(
    train_dataset,
    config,
    device,
    prompt_template="a photo of X X",
):

    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    allattrs = train_dataset.attrs 
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    tokenized = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attributes + classes
        ]
    )
    orig_token_embedding = clip_model.token_embedding(tokenized.to(device))
    soft_embedding = torch.zeros(
        (len(attributes) + len(classes), orig_token_embedding.size(-1)),
    ) #
    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized[idx].argmax()
        soft_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        if config.train_prefix:
            ctx_init = "a photo of "
            n_ctx = len(ctx_init.split())
            tokenized_prompt = clip.tokenize(ctx_init,
                                context_length=config.context_length).to(device)
            with torch.no_grad():
                orig_token_embedding_prefix = clip_model.token_embedding(tokenized_prompt.to(device))
            ctx_vectors = orig_token_embedding_prefix[0, 1 : 1 + n_ctx, :]
            soft_embeddings_prefix = nn.Parameter(ctx_vectors)
        else:
            soft_embeddings_prefix = None

    soft_embedding = nn.Parameter(soft_embedding)

    class_token_ids = clip.tokenize(
        [prompt_template],
        context_length=config.context_length,
    )
    offset = len(attributes)

    return (
        clip_model,
        soft_embedding,
        soft_embeddings_prefix,

        class_token_ids,
        offset
    )


class HPLInterface(CLIPInterfaceHie):
    def __init__(
        self,
        clip_model,
        config: argparse.ArgumentParser,
        offset,
        soft_embeddings: torch.nn.Parameter,
        soft_embeddings_hpl: torch.nn.Parameter,
        soft_embeddings_prefix: torch.nn.Parameter or None,
        img_transform,
        class_token_ids: torch.tensor,
        device: torch.device = "cuda:0",
        enable_pos_emb=True,
        enable_img_transform=1,
        attr_dropout=0.0,
    ):
        super().__init__(
            clip_model,
            config,
            class_token_ids,
            soft_embeddings,
            soft_embeddings_hpl,
            soft_embeddings_prefix,
            img_transform,
            device=device,
            enable_pos_emb=enable_pos_emb,
            enable_img_transform=enable_img_transform,
        )

        self.offset = offset
        self.attr_dropout = nn.Dropout(attr_dropout)

    def construct_token_tensors(self, pair_idx, attrs, objs):
        """Function creates the token tensor for further inference.

        Args:
            pair_idx (torch.Tensor): Shape [N x 2], where N is the number
                of pairs of attr and obj

        Returns:
            torch.Tensor: token tensor passed to the text encoder;
                shape [N x context_length x 512]
        """
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        class_token_ids_attr = self.token_ids.repeat(len(attrs), 1)
        class_token_ids_obj = self.token_ids.repeat(len(objs), 1)
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)
        token_tensor_attr = self.clip_model.token_embedding(
            class_token_ids_attr.to(self.device)
        ).type(self.clip_model.dtype)
        token_tensor_obj = self.clip_model.token_embedding(
            class_token_ids_obj.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.token_ids[0].argmax())
        soft_embeddings = self.attr_dropout(self.soft_embeddings)
        soft_embeddings_hpl = self.attr_dropout(self.soft_embeddings_hpl)
        token_tensor[:, eos_idx - 2, :] = soft_embeddings[
            attr_idx
        ].type(self.clip_model.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_embeddings[
            obj_idx + self.offset
        ].type(self.clip_model.dtype)

        token_tensor_attr[:, eos_idx - 2, :] = soft_embeddings_hpl[
            attrs
        ].type(self.clip_model.dtype)
        token_tensor_attr[:, eos_idx - 1, :] = soft_embeddings_hpl[
            -1
        ].type(self.clip_model.dtype)

        token_tensor_obj[:, eos_idx - 2, :] = soft_embeddings_hpl[
            -2
        ].type(self.clip_model.dtype)
        token_tensor_obj[:, eos_idx - 1, :] = soft_embeddings_hpl[
            objs + self.offset
        ].type(self.clip_model.dtype)
        
        if self.config.train_prefix:
            token_tensor[:, 1 : len(self.soft_embeddings_prefix) + 1, :] = self.soft_embeddings_prefix.type(self.clip_model.dtype)
            
            token_tensor_attr[:, 1 : len(self.soft_embeddings_prefix) + 1, :] = self.soft_embeddings_prefix.type(self.clip_model.dtype)
            
            token_tensor_obj[:, 1 : len(self.soft_embeddings_prefix) + 1, :] = self.soft_embeddings_prefix.type(self.clip_model.dtype)
        
        return token_tensor, token_tensor_attr, token_tensor_obj

    def construct_token_tensors_pair(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.token_ids[0].argmax())
        soft_embeddings = self.attr_dropout(self.soft_embeddings)
        token_tensor[:, eos_idx - 2, :] = soft_embeddings[
            attr_idx
        ].type(self.clip_model.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_embeddings[
            obj_idx + self.offset
        ].type(self.clip_model.dtype)
        if self.config.train_prefix:
            token_tensor[:, 1 : len(self.soft_embeddings_prefix) + 1, :] = self.soft_embeddings_prefix.type(self.clip_model.dtype)
        return token_tensor

    def construct_token_tensors_attr(self, attrs):
        class_token_ids_attr = self.token_ids.repeat(len(attrs), 1)
        token_tensor_attr = self.clip_model.token_embedding(
            class_token_ids_attr.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.token_ids[0].argmax())
        soft_embeddings_hpl = self.attr_dropout(self.soft_embeddings_hpl)

        token_tensor_attr[:, eos_idx - 2, :] = soft_embeddings_hpl[
            attrs
        ].type(self.clip_model.dtype)
        token_tensor_attr[:, eos_idx - 1, :] = soft_embeddings_hpl[
            -1
        ].type(self.clip_model.dtype)
        if self.config.train_prefix:
            token_tensor_attr[:, 1 : len(self.soft_embeddings_prefix) + 1, :] = self.soft_embeddings_prefix.type(self.clip_model.dtype)
        return token_tensor_attr

    def construct_token_tensors_obj(self, objs):
        class_token_ids_obj = self.token_ids.repeat(len(objs), 1)
        token_tensor_obj = self.clip_model.token_embedding(
            class_token_ids_obj.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.token_ids[0].argmax())
        soft_embeddings_hpl = self.attr_dropout(self.soft_embeddings_hpl)

        token_tensor_obj[:, eos_idx - 2, :] = soft_embeddings_hpl[
            -2
        ].type(self.clip_model.dtype)
        token_tensor_obj[:, eos_idx - 1, :] = soft_embeddings_hpl[
            objs + self.offset
        ].type(self.clip_model.dtype)
        if self.config.train_prefix:
            token_tensor_obj[:, 1 : len(self.soft_embeddings_prefix) + 1, :] = self.soft_embeddings_prefix.type(self.clip_model.dtype)
        return token_tensor_obj

def csp_init_hpl(
    train_dataset,
    config,
    device,
    prompt_template="a photo of X X",
):

    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]


    tokenized = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attributes + classes
        ]
    )
    tokenized_hpl = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attributes + classes + ['the', 'thing']  # or maybe 'a'
        ]
    )

    orig_token_embedding = clip_model.token_embedding(tokenized.to(device))
    orig_token_embedding_hpl = clip_model.token_embedding(tokenized_hpl.to(device))

    soft_embedding = torch.zeros(
        (len(attributes) + len(classes), orig_token_embedding.size(-1)),
    )
    soft_embedding_hpl = torch.zeros(
        (len(attributes) + len(classes) + 2, orig_token_embedding.size(-1)),
    )

    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized[idx].argmax()
        soft_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
    for idx, rep in enumerate(orig_token_embedding_hpl):
        eos_idx = tokenized_hpl[idx].argmax()
        soft_embedding_hpl[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

    if config.train_prefix:
        ctx_init = "a photo of "
        n_ctx = len(ctx_init.split())
        tokenized_prompt = clip.tokenize(ctx_init,
                            context_length=config.context_length).to(device)
        with torch.no_grad():
            orig_token_embedding_prefix = clip_model.token_embedding(tokenized_prompt.to(device))
        ctx_vectors = orig_token_embedding_prefix[0, 1 : 1 + n_ctx, :]
        soft_embeddings_prefix = nn.Parameter(ctx_vectors)
    else:
        soft_embeddings_prefix = None


    soft_embedding = nn.Parameter(soft_embedding)
    soft_embedding_hpl = nn.Parameter(soft_embedding_hpl)

    layers = [512, 768]  # todo hyperparams below
    img_embedder_pair = MLP(1024, 1024, num_layers=2, relu=False, bias=True, dropout=True, norm=True, layers=layers)
    img_embedder_attr = MLP(1024, 1024, num_layers=2, relu=False, bias=True, dropout=True, norm=True, layers=layers)
    img_embedder_obj = MLP(1024, 1024, num_layers=2, relu=False, bias=True, dropout=True, norm=True, layers=layers)

    img_transform = nn.ModuleDict(
        {
            'pair': img_embedder_pair,
            'attr': img_embedder_attr,
            'obj': img_embedder_obj,
        }
    )

    class_token_ids = clip.tokenize(
        [prompt_template],
        context_length=config.context_length,
    )
    offset = len(attributes)

    return (
        clip_model,
        soft_embedding,
        soft_embedding_hpl,
        soft_embeddings_prefix,
        img_transform,
        class_token_ids,
        offset
    )


def get_hpl(train_dataset, config, device):

    (
        clip_model,
        soft_embedding,
        soft_embedding_hpl,
        soft_embeddings_prefix,
        img_transform,
        class_token_ids,
        offset
    ) = csp_init_hpl(train_dataset, config, device)
    
    params = [
        {"params": soft_embedding},
        {"params": soft_embedding_hpl},
        {"params": img_transform.parameters()},
    ]

    if config.train_prefix:
        params.append({"params": soft_embeddings_prefix})

    optimizer = torch.optim.Adam(
        params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    interface = HPLInterface(
        clip_model,
        config,
        offset,
        soft_embedding,
        soft_embedding_hpl,
        soft_embeddings_prefix,
        img_transform,
        class_token_ids,
        device,
        enable_img_transform=config.enable_img_transform,
        attr_dropout=config.attr_dropout
    )

    return interface, optimizer


def get_csp(train_dataset, config, device):

    (
        clip_model,
        soft_embedding,
        soft_embeddings_prefix,
        class_token_ids,
        offset
    ) = csp_init(train_dataset, config, device)

    params = [
        {"params": soft_embedding},
    ]
    if config.train_prefix:
        params.append({"params": soft_embeddings_prefix})
    optimizer = torch.optim.Adam(
        params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    interface = CSPInterface(
        clip_model,
        config,
        offset,
        soft_embedding,
        soft_embeddings_prefix,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout
    )
    return interface, optimizer


class MLP(nn.Module):
    """
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    """

    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers.pop(0)
            mod.append(nn.Linear(incoming, outgoing, bias=bias))

            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p=0.5))

        mod.append(nn.Linear(incoming, out_dim, bias=bias))

        if relu:
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.1))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        return self.mod(x)
