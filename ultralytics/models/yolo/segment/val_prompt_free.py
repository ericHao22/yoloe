# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Prompt-free validator.

Workflow per batch:
1) Use prompt image for prompt-free detection to get class ids
2) Build text prompts from detected classes
3) Set model classes for downstream prediction on target image
"""

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from ultralytics import YOLOE
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.models.yolo.yoloe.val import YOLOEValidatorMixin
from ultralytics.nn.text_model import build_text_model
from ultralytics.utils import LOGGER


class PromptFreeSegmentationValidator(YOLOEValidatorMixin, SegmentationValidator):
    """Segmentation validator that injects prompt-free class priors each batch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pf_model = None
        self.tpe_model = build_text_model(self.args.get("text_model"), device=self.device)
        self.max_prompt_classes = getattr(self.args, "max_prompt_classes", 80)
        self.similarity_threshold = getattr(self.args, "similarity_threshold", 0.8)
        self.target_classes = list(['person', 'dog'])
        self.allowed_classes = None  # per-batch filter (keeps metrics mapping stable)

    def init_metrics(self, model):
        super().init_metrics(model)
        # lazily load prompt-free model
        model_name = Path(model.model.pt_path).stem
        LOGGER.info(f'Loading prompt-free model for {model_name}...')
        path = hf_hub_download(repo_id="jameslahm/yoloe", filename=f"{model_name}-pf.pt")
        pf_model = YOLOE(path)
        pf_model.to(self.device)
        pf_model.eval()
        
        # build vocab
        unfused_model = YOLOE("yoloe-v8l.yaml")
        unfused_model.load(model.model.pt_path)
        unfused_model.eval()
        unfused_model.cuda()

        with open('tools/ram_tag_list.txt', 'r') as f:
            names = [x.strip() for x in f.readlines()]
        vocab = unfused_model.get_vocab(names)

        print(f'Setting vocab with {len(names)} names')

        pf_model.set_vocab(vocab, names=names)
        pf_model.model.model[-1].is_fused = True
        pf_model.model.model[-1].conf = 0.001
        pf_model.model.model[-1].max_det = 1000

        self.pf_model = pf_model

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if "prompt_img" not in batch:
            self.allowed_classes = None
            return batch
        # infer prompt classes
        batch["prompt_img"] = batch["prompt_img"].to(self.device)

        if batch["prompt_img"].dtype == torch.uint8 or batch["prompt_img"].max() > 1:
            batch["prompt_img"] = batch["prompt_img"].float() / 255.0

        prompt_classes = self._infer_prompt_classes(batch["prompt_img"])

        if prompt_classes:
            filtered_classes = self._filter_by_similarity(prompt_classes, self.target_classes)
            self.allowed_classes = filtered_classes if filtered_classes else None
        else:
            LOGGER.warning('No classes detected in prompt image; skipping setting classes.')
            self.allowed_classes = None
        return batch

    @torch.no_grad()
    def _infer_prompt_classes(self, prompt_img: torch.Tensor) -> List[str]:
        out = self.pf_model.predict(prompt_img, conf=0.25, iou=0.7, max_det=1000, verbose=False, imgsz=640)
        all_classes = set()
        for result in out:
            boxes = result.boxes
            if boxes is not None and len(boxes):
                class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
                for cls_id in class_ids:
                    all_classes.add(self.pf_model.model.names[cls_id])
            if len(all_classes) >= self.max_prompt_classes:
                break
        prompt_classes = list(all_classes)
        LOGGER.info(f'Prompt-free detected classes: {prompt_classes}')
        return prompt_classes

    @torch.no_grad()
    def _get_classes_embeddings(self, class_names) -> torch.Tensor:
        """Get text embeddings for given class names using the CLIP model."""
        if not class_names:
            return None
        text_token = self.tpe_model.tokenize(class_names)
        txt_feats = self.tpe_model.encode_text(text_token)
        txt_feats = txt_feats.reshape(-1, len(class_names), txt_feats.shape[-1])
        return txt_feats

    @torch.no_grad()
    def _filter_by_similarity(self, prompt_classes: List[str], target_classes: List[str]) -> List[str]:
        """Filter target classes based on text embedding similarity to prompt classes.
        
        Args:
            prompt_classes: List of class names detected from prompt image
            target_classes: List of all available target class names
            similarity_threshold: Only keep target classes with max similarity >= threshold
            
        Returns:
            Filtered list of target class names with high similarity to prompt classes
        """
        if not prompt_classes or not target_classes:
            return prompt_classes
        
        try:
            # Get embeddings for both prompt and target classes
            prompt_embeddings = self._get_classes_embeddings(prompt_classes)  # (1, len(prompt), 512)
            target_embeddings = self._get_classes_embeddings(target_classes)  # (1, len(target), 512)
            
            # Normalize embeddings
            prompt_emb = F.normalize(prompt_embeddings.squeeze(0), dim=-1)  # (len(prompt), 512)
            target_emb = F.normalize(target_embeddings.squeeze(0), dim=-1)  # (len(target), 512)
            
            # Compute similarity: (len(target), len(prompt))
            similarity = torch.matmul(target_emb, prompt_emb.t())  # cosine similarity
            
            # Keep target classes with max similarity >= threshold
            max_similarity, _ = torch.max(similarity, dim=1)
            filtered_indices = (max_similarity >= self.similarity_threshold).nonzero(as_tuple=True)[0]
            
            filtered_classes = [target_classes[i] for i in filtered_indices.cpu().numpy()]
            
            LOGGER.info(
                f'Filtered {len(filtered_classes)}/{len(target_classes)} target classes '
                f'with similarity >= {self.similarity_threshold}'
            )
            
            return filtered_classes if filtered_classes else list(prompt_classes)
        
        except Exception as e:
            LOGGER.warning(f'Similarity filtering failed: {e}; using prompt classes directly')
            return prompt_classes

    def postprocess(self, preds):
        p, proto = super().postprocess(preds)

        if not getattr(self, "allowed_classes", None):
            return p, proto

        # map class names to ids in the fixed dataset order
        name_to_id = (
            {v: k for k, v in self.names.items()}
            if isinstance(self.names, dict)
            else {n: i for i, n in enumerate(self.names)}
        )
        # select a proper device for keep_ids
        device = p[0].device if isinstance(p, (list, tuple)) and len(p) and isinstance(p[0], torch.Tensor) else self.device
        keep_ids = torch.tensor([name_to_id[c] for c in self.allowed_classes if c in name_to_id], device=device)

        # filter per-image predictions by class column (index 5)
        for i in range(len(p)):
            pred = p[i]
            if pred is None or pred.numel() == 0:
                continue
            if keep_ids.numel() == 0:
                p[i] = pred[0:0]  # no classes to keep, return empty tensor
                continue
            keep = torch.isin(pred[:, 5].to(keep_ids.device), keep_ids)
            p[i] = pred[keep]

        return p, proto
