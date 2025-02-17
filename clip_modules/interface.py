import argparse

import torch
import torch.nn as nn
from clip.model import CLIP

from .text_encoder import CustomTextEncoder


class CLIPInterfaceHie(torch.nn.Module):
    def __init__(
        self,
        clip_model: CLIP,
        config: argparse.ArgumentParser,
        token_ids: torch.tensor,
        soft_embeddings: torch.nn.Parameter = None,
        soft_embeddings_hpl: torch.nn.Parameter = None,
        soft_embeddings_prefix: torch.nn.Parameter = None,
        img_transform: torch.nn.ModuleDict = None,
        dtype: torch.dtype = None,
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = False,
        enable_img_transform: int = 0,
    ):
        """CLIP interface for our custom modules.

        Args:
            clip_model (CLIP): the clip model
            config (argparse.ArgumentParser): arguments used for
                training
            token_ids (torch.tensor): the input token ids to the text
                encoder
            soft_embeddings (torch.nn.Parameter, optional): the only
                parameter that we finetune in the experiment.
                Defaults to None.
            dtype (torch.dtype, optional): torch dtype for the
                transformer. This allows the half precision option.
                Defaults to None.
            device (torch.device, optional): the device where the model
                should be loaded. Defaults to "cuda:0".
            enable_pos_emb (bool, optional): if true, adds the learned
                positional embeddings. Defaults to False.
        """
        super().__init__()

        self.config = config

        self.clip_model = clip_model

        if dtype is None and device == "cpu":
            self.dtype = torch.float32
        elif dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype

        self.device = device

        self.enable_pos_emb = enable_pos_emb
        self.enable_img_transform = enable_img_transform
        print(f'enable_img_transform: {bool(enable_img_transform)}')

        self.text_encoder = CustomTextEncoder(clip_model, self.dtype)
        for params in self.text_encoder.parameters():
            params.requires_grad = False
        self.clip_model.text_projection.requires_grad = False

        self.token_ids = token_ids
        self.soft_embeddings = soft_embeddings
        self.soft_embeddings_hpl = soft_embeddings_hpl
        self.soft_embeddings_prefix = soft_embeddings_prefix
        self.img_transform = img_transform.to(device)
        # self.img_transform = img_transform.type(self.dtype).to(device)

    def encode_image(self, imgs):
        return self.clip_model.encode_image(imgs)

    def encode_text(self, text, enable_pos_emb=True):
        return self.text_encoder.encode_text(
            text, enable_pos_emb=enable_pos_emb
        )

    def tokenize(self, text):
        return self.text_encoder.tokenize(text)

    def set_soft_embeddings(self, se, se_hpl, se_prefix=None):
        if (se.shape == self.soft_embeddings.shape) and (se_hpl.shape == self.soft_embeddings_hpl.shape):
            self.state_dict()['soft_embeddings'].copy_(se)
            self.state_dict()['soft_embeddings_hpl'].copy_(se_hpl)
            if se_prefix is not None:
                self.state_dict()['soft_embeddings_prefix'].copy_(se_prefix)
        else:
            raise RuntimeError(f"Error: Incorrect Soft Embedding Shape {se.shape}, Expecting {self.soft_embeddings.shape}!"
                               f"Error: Incorrect Soft Embedding Shape {se_hpl.shape}, Expecting {self.soft_embeddings_hpl.shape}!")

    def construct_token_tensors(self, idx, idx_1, idx_2):
        """The function is used to generate token tokens. These
        token tensors can be None or custom. For custom token_tensors
        the class needs to be inherited and the function should be
        replaced.

        Raises:
            NotImplementedError: raises error if the model contains
            soft embeddings but does not make custom modifications.

        Returns:
            torch.Tensor: returns torch.Tensor or None
        """
        if self.soft_embeddings is None:
            return None
        else:
            # Implement a custom version
            raise NotImplementedError

    def forward(self, batch_img, idx, idx_attr, idx_obj):
        batch_img = batch_img.to(self.device)  # bs x 1024

        token_tensors, token_tensors_attr, token_tensors_obj = self.construct_token_tensors(idx, idx_attr, idx_obj)

        text_features = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        )
        text_features_attr = self.text_encoder(
            self.token_ids,
            token_tensors_attr,
            enable_pos_emb=self.enable_pos_emb,
        )
        text_features_obj = self.text_encoder(
            self.token_ids,
            token_tensors_obj,
            enable_pos_emb=self.enable_pos_emb,
        )

        #_text_features = text_features[idx, :]
        _text_features = text_features
        _text_features_attr = text_features_attr
        _text_features_obj = text_features_obj

        idx_text_features = _text_features / _text_features.norm(
            dim=-1, keepdim=True
        )
        idx_text_features_attr = _text_features_attr / _text_features_attr.norm(
            dim=-1, keepdim=True
        )
        idx_text_features_obj = _text_features_obj / _text_features_obj.norm(
            dim=-1, keepdim=True
        )
        # todo add 3 transformations for img
        if self.enable_img_transform:
            batch_img_pair = self.img_transform['pair'].type(self.dtype)(batch_img)
            batch_img_attr = self.img_transform['attr'].type(self.dtype)(batch_img)
            batch_img_obj = self.img_transform['obj'].type(self.dtype)(batch_img)
            normalized_img_pair = batch_img_pair / batch_img_pair.norm(dim=-1, keepdim=True)
            normalized_img_attr = batch_img_attr / batch_img_attr.norm(dim=-1, keepdim=True)
            normalized_img_obj = batch_img_obj / batch_img_obj.norm(dim=-1, keepdim=True)
        else:
            normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
            normalized_img_pair = normalized_img
            normalized_img_attr = normalized_img
            normalized_img_obj = normalized_img
        logits = (
            self.clip_model.logit_scale.exp()
            * normalized_img_pair
            @ idx_text_features.t()
        )
        logits_attr = (
                self.clip_model.logit_scale.exp()
                * normalized_img_attr
                @ idx_text_features_attr.t()
        )
        logits_obj = (
                self.clip_model.logit_scale.exp()
                * normalized_img_obj
                @ idx_text_features_obj.t()
        )

        return logits, logits_attr, logits_obj



class CLIPInterface(torch.nn.Module):
    def __init__(
        self,
        clip_model: CLIP,
        config: argparse.ArgumentParser,
        token_ids: torch.tensor,
        soft_embeddings: torch.nn.Parameter = None,
        soft_embeddings_prefix: torch.nn.Parameter = None,
        dtype: torch.dtype = None,
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = False,
    ):
        """CLIP interface for our custom modules.

        Args:
            clip_model (CLIP): the clip model
            config (argparse.ArgumentParser): arguments used for
                training
            token_ids (torch.tensor): the input token ids to the text
                encoder
            soft_embeddings (torch.nn.Parameter, optional): the only
                parameter that we finetune in the experiment.
                Defaults to None.
            dtype (torch.dtype, optional): torch dtype for the
                transformer. This allows the half precision option.
                Defaults to None.
            device (torch.device, optional): the device where the model
                should be loaded. Defaults to "cuda:0".
            enable_pos_emb (bool, optional): if true, adds the learned
                positional embeddings. Defaults to False.
        """
        super().__init__()

        self.config = config

        self.clip_model = clip_model

        if dtype is None and device == "cpu":
            self.dtype = torch.float32
        elif dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype

        self.device = device

        self.enable_pos_emb = enable_pos_emb

        self.text_encoder = CustomTextEncoder(clip_model, self.dtype)
        for params in self.text_encoder.parameters():
            params.requires_grad = False
        self.clip_model.text_projection.requires_grad = False

        self.token_ids = token_ids
        self.soft_embeddings = soft_embeddings
        self.soft_embeddings_prefix = soft_embeddings_prefix

    def encode_image(self, imgs):
        return self.clip_model.encode_image(imgs)

    def encode_text(self, text, enable_pos_emb=True):
        return self.text_encoder.encode_text(
            text, enable_pos_emb=enable_pos_emb
        )

    def tokenize(self, text):
        return self.text_encoder.tokenize(text)

    def set_soft_embeddings(self, se, se_prefix=None):
        if se.shape == self.soft_embeddings.shape:
            self.state_dict()['soft_embeddings'].copy_(se)
            if se_prefix is not None:
                self.state_dict()['soft_embeddings_prefix'].copy_(se_prefix)
        else:
            raise RuntimeError(f"Error: Incorrect Soft Embedding Shape {se.shape}, Expecting {self.soft_embeddings.shape}!")

    def construct_token_tensors(self, idx):
        """The function is used to generate token tokens. These
        token tensors can be None or custom. For custom token_tensors
        the class needs to be inherited and the function should be
        replaced.

        Raises:
            NotImplementedError: raises error if the model contains
            soft embeddings but does not make custom modifications.

        Returns:
            torch.Tensor: returns torch.Tensor or None
        """
        if self.soft_embeddings is None:
            return None
        else:
            # Implement a custom version
            raise NotImplementedError

    def forward(self, batch_img, idx):
        batch_img = batch_img.to(self.device)

        token_tensors = self.construct_token_tensors(idx) 

        text_features = self.text_encoder(
            self.token_ids, 
            token_tensors, 
            enable_pos_emb=self.enable_pos_emb,
        ) 

        #_text_features = text_features[idx, :]
        _text_features = text_features 

        idx_text_features = _text_features / _text_features.norm(
            dim=-1, keepdim=True
        )
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        logits = (
            self.clip_model.logit_scale.exp()
            * normalized_img 
            @ idx_text_features.t()
        )

        return logits


