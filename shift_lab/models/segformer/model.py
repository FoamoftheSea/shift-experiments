from dataclasses import dataclass
from typing import Optional, List, Union, Tuple, Dict

import torch
from shift_lab.models.segformer.metrics import DepthTrainLoss
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import SegformerForSemanticSegmentation, GLPNConfig
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.models.glpn.modeling_glpn import GLPNDecoder, GLPNDepthEstimationHead
from transformers.utils import logging

from shift_lab.models.segformer.constants import SegformerTask

logger = logging.get_logger(__name__)


@dataclass
class MultitaskSegformerOutput(SemanticSegmenterOutput):
    loss: Optional[Dict[str, torch.FloatTensor]] = None
    depth_pred: Optional[torch.FloatTensor] = None


class MultitaskSegformer(SegformerForSemanticSegmentation):

    def __init__(
        self,
        config,
        do_reduce_labels: Optional[bool] = None,
        tasks: Optional[List[SegformerTask]] = None,
        **kwargs
    ):
        self.class_loss_weights = None
        if not hasattr(config, "do_reduce_labels"):
            config.do_reduce_labels = True if do_reduce_labels is None else do_reduce_labels
        else:
            if do_reduce_labels is not None and do_reduce_labels != config.do_reduce_labels:
                logger.warning("'do_reduce_labels' setting passed but conflicts with the setting in pretrained model.")
                logger.warning("Defaulting to setting in pretrained config to avoid class ID conflict.")
        self.tasks = tasks if tasks is not None else [SegformerTask.SEMSEG, SegformerTask.DEPTH]

        if SegformerTask.DEPTH in self.tasks:
            if not hasattr(config, "depth_config") or config.depth_config is None:
                config.depth_config = GLPNConfig(
                    num_channels=config.num_channels,
                    num_encoder_blocks=config.num_encoder_blocks,
                    depths=config.depths,
                    sr_ratios=config.sr_ratios,
                    hidden_sizes=config.hidden_sizes,
                    patch_sizes=config.patch_sizes,
                    strides=config.strides,
                    num_attention_heads=config.num_attention_heads,
                    mlp_ratios=config.mlp_ratios,
                    hidden_act=config.hidden_act,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    drop_path_rate=config.drop_path_rate,
                    layer_norm_eps=config.layer_norm_eps,
                    decoder_hidden_size=64,
                    max_depth=10,
                    head_in_index=-1,
                )

            # else:
            #     config.depth_config = GLPNConfig(**config.depth_config)

        super().__init__(config)

        if hasattr(config, "depth_config") and config.depth_config is not None:
            if isinstance(config.depth_config, dict):
                config.depth_config = GLPNConfig(**config.depth_config)
            config.depth_config.silog_lambda = kwargs.get(
                "silog_lambda",
                config.depth_config.__dict__.get("silog_lambda", 0.25)
            )
            self.depth_decoder = GLPNDecoder(config.depth_config)
            self.depth_head = GLPNDepthEstimationHead(config.depth_config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels_semantic: Optional[torch.LongTensor] = None,
        labels_depth: Optional = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states)
        if SegformerTask.DEPTH in self.tasks:
            depth_decoder_out = self.depth_decoder(encoder_hidden_states)
            predicted_depth = self.depth_head(depth_decoder_out)
        else:
            predicted_depth = None

        loss = {}
        if labels_semantic is not None:
            # upsample logits to the images' original size
            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=labels_semantic.shape[-2:], mode="bilinear", align_corners=False
            )
            if self.config.num_labels > 1:
                loss_fct = CrossEntropyLoss(
                    ignore_index=self.config.semantic_loss_ignore_index,
                    weight=self.class_loss_weights
                )
                labels_loss = loss_fct(upsampled_logits, labels_semantic)
            elif self.config.num_labels == 1:
                valid_mask = ((labels_semantic >= 0) & (labels_semantic != self.config.semantic_loss_ignore_index)).float()
                loss_fct = BCEWithLogitsLoss(reduction="none")
                labels_loss = loss_fct(upsampled_logits.squeeze(1), labels_semantic.float())
                labels_loss = (labels_loss * valid_mask).mean()
            else:
                raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")

            loss["semseg"] = labels_loss

        if SegformerTask.DEPTH in self.tasks and labels_depth is not None:
            loss_fct = DepthTrainLoss(silog_lambda=self.config.depth_config.silog_lambda)
            # Labels are converted to log by loss function, model inference is in log depth
            loss["depth"] = loss_fct(predicted_depth, labels_depth)

        if len(loss) == 0:
            loss = None

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultitaskSegformerOutput(
            loss=loss,
            logits=logits,
            depth_pred=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
