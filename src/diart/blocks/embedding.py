# from typing import Optional, Union, Text

# import torch
# from einops import rearrange

# from .. import functional as F
# from ..features import TemporalFeatures, TemporalFeatureFormatter
# from ..models import EmbeddingModel

# # from espnet2.bin.spk_inference import Speech2Embedding


# class SpeakerEmbedding:
#     def __init__(self, model: EmbeddingModel, device: Optional[torch.device] = None, emb_model = None):
#         self.emb_model = emb_model
#         if self.emb_model == "spkrnet":
#             import nemo.collections.asr as nemo_asr
#             self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="speakerverification_speakernet")
#             self.device = device
#             if self.device is None:
#                 self.device = torch.device("cpu")
#             self.waveform_formatter = TemporalFeatureFormatter()
#             self.weights_formatter = TemporalFeatureFormatter()
#         else: 
#             self.model = model
#             self.model.eval()
#             self.device = device
#             if self.device is None:
#                 self.device = torch.device("cpu")
#             self.model.to(self.device)
#             self.waveform_formatter = TemporalFeatureFormatter()
#             self.weights_formatter = TemporalFeatureFormatter()

#     @staticmethod
#     def from_pretrained(
#         model,
#         use_hf_token: Union[Text, bool, None] = True,
#         device: Optional[torch.device] = None,
#     ) -> "SpeakerEmbedding":
#         emb_model = EmbeddingModel.from_pretrained(model, use_hf_token)
#         return SpeakerEmbedding(emb_model, device)

#     def __call__(
#         self, waveform: TemporalFeatures, weights: Optional[TemporalFeatures] = None
#     ) -> torch.Tensor:
#         """
#         Calculate speaker embeddings of input audio.
#         If weights are given, calculate many speaker embeddings from the same waveform.

#         Parameters
#         ----------
#         waveform: TemporalFeatures, shape (samples, channels) or (batch, samples, channels)
#         weights: Optional[TemporalFeatures], shape (frames, speakers) or (batch, frames, speakers)
#             Per-speaker and per-frame weights. Defaults to no weights.

#         Returns
#         -------
#         embeddings: torch.Tensor
#             If weights are provided, the shape is (batch, speakers, embedding_dim),
#             otherwise the shape is (batch, embedding_dim).
#             If batch size == 1, the batch dimension is omitted.
#         """
#         with torch.no_grad():
#             inputs = self.waveform_formatter.cast(waveform).to(self.device)
#             inputs = rearrange(inputs, "batch sample channel -> batch channel sample")
#             if weights is not None and not self.emb_model:
#                 weights = self.weights_formatter.cast(weights).to(self.device)
#                 batch_size, _, num_speakers = weights.shape
#                 inputs = inputs.repeat(1, num_speakers, 1)
#                 weights = rearrange(weights, "batch frame spk -> (batch spk) frame")
#                 inputs = rearrange(inputs, "batch spk sample -> (batch spk) 1 sample")
#                 output = rearrange(
#                     self.model(inputs, weights),
#                     "(batch spk) feat -> batch spk feat",
#                     batch=batch_size,
#                     spk=num_speakers,
#                 )
#             elif weights is not None and self.emb_model:
#                 weights = self.weights_formatter.cast(weights).to(self.device)
#                 batch_size, _, num_speakers = weights.shape
#                 inputs = inputs.repeat(1, num_speakers, 1)
#                 weights = rearrange(weights, "batch frame spk -> (batch spk) frame")
#                 inputs = rearrange(inputs, "batch spk sample -> (batch spk) 1 sample")
#                 output = []
#                 inputs = inputs.squeeze(dim=1)
#                 # print("Inputs shape: ", inputs.shape)
#                 for i in inputs:
#                     # ind_input = i.squeeze()
#                     ind_output = self.model.get_embedding(i)
#                     output.append(ind_output)
#                 # output = self.model.get_embedding(inputs)
#                 output = torch.stack(output)
#                 output = output.squeeze(dim=1)
#                 return output.cpu()
#             else:
#                 output = self.model(inputs)
#             return output.squeeze().cpu()


# class OverlappedSpeechPenalty:
#     """Applies a penalty on overlapping speech and low-confidence regions to speaker segmentation scores.

#     .. note::
#         For more information, see `"Overlap-Aware Low-Latency Online Speaker Diarization
#         based on End-to-End Local Segmentation" <https://github.com/juanmc2005/diart/blob/main/paper.pdf>`_
#         (Section 2.2.1 Segmentation-driven speaker embedding). This block implements Equation 2.

#     Parameters
#     ----------
#     gamma: float, optional
#         Exponent to lower low-confidence predictions.
#         Defaults to 3.
#     beta: float, optional
#         Temperature parameter (actually 1/beta) to lower joint speaker activations.
#         Defaults to 10.
#     normalize: bool, optional
#         Whether to min-max normalize weights to be in the range [0, 1].
#         Defaults to False.
#     """

#     def __init__(self, gamma: float = 3, beta: float = 10, normalize: bool = False):
#         self.gamma = gamma
#         self.beta = beta
#         self.formatter = TemporalFeatureFormatter()
#         self.normalize = normalize

#     def __call__(self, segmentation: TemporalFeatures) -> TemporalFeatures:
#         weights = self.formatter.cast(segmentation)  # shape (batch, frames, speakers)
#         with torch.inference_mode():
#             weights = F.overlapped_speech_penalty(weights, self.gamma, self.beta)
#             if self.normalize:
#                 min_values = weights.min(dim=1, keepdim=True).values
#                 max_values = weights.max(dim=1, keepdim=True).values
#                 weights = (weights - min_values) / (max_values - min_values)
#                 weights.nan_to_num_(1e-8)
#         return self.formatter.restore_type(weights)


# class EmbeddingNormalization:
#     def __init__(self, norm: Union[float, torch.Tensor] = 1):
#         self.norm = norm
#         # Add batch dimension if missing
#         if isinstance(self.norm, torch.Tensor) and self.norm.ndim == 2:
#             self.norm = self.norm.unsqueeze(0)

#     def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
#         with torch.inference_mode():
#             norm_embs = F.normalize_embeddings(embeddings, self.norm)
#         return norm_embs


# class OverlapAwareSpeakerEmbedding:
#     """
#     Extract overlap-aware speaker embeddings given an audio chunk and its segmentation.

#     Parameters
#     ----------
#     model: EmbeddingModel
#         A pre-trained embedding model.
#     gamma: float, optional
#         Exponent to lower low-confidence predictions.
#         Defaults to 3.
#     beta: float, optional
#         Softmax's temperature parameter (actually 1/beta) to lower joint speaker activations.
#         Defaults to 10.
#     norm: float or torch.Tensor of shape (batch, speakers, 1) where batch is optional
#         The target norm for the embeddings. It can be different for each speaker.
#         Defaults to 1.
#     normalize_weights: bool, optional
#         Whether to min-max normalize embedding weights to be in the range [0, 1].
#     device: Optional[torch.device]
#         The device on which to run the embedding model.
#         Defaults to GPU if available or CPU if not.
#     """

#     def __init__(
#         self,
#         model: EmbeddingModel,
#         gamma: float = 3,
#         beta: float = 10,
#         norm: Union[float, torch.Tensor] = 1,
#         normalize_weights: bool = False,
#         device: Optional[torch.device] = None,
#     ):
#         self.embedding = SpeakerEmbedding(model, device)
#         self.osp = OverlappedSpeechPenalty(gamma, beta, normalize_weights)
#         self.normalize = EmbeddingNormalization(norm)

#     @staticmethod
#     def from_pretrained(
#         model,
#         gamma: float = 3,
#         beta: float = 10,
#         norm: Union[float, torch.Tensor] = 1,
#         use_hf_token: Union[Text, bool, None] = True,
#         normalize_weights: bool = False,
#         device: Optional[torch.device] = None,
#     ):
#         model = EmbeddingModel.from_pretrained(model, use_hf_token)
#         return OverlapAwareSpeakerEmbedding(
#             model, gamma, beta, norm, normalize_weights, device
#         )

#     def __call__(
#         self, waveform: TemporalFeatures, segmentation: TemporalFeatures
#     ) -> torch.Tensor:
#         return self.normalize(self.embedding(waveform, self.osp(segmentation)))
    
# class SpkrNetSpeakerEmbedding:
#     def __init__(
#         self,
#         model: EmbeddingModel,
#         gamma: float = 3,
#         beta: float = 10,
#         norm: Union[float, torch.Tensor] = 1,
#         normalize_weights: bool = False,
#         device: Optional[torch.device] = None,
#     ):
#         self.embedding = SpeakerEmbedding(model, device, emb_model="spkrnet")
#         self.osp = OverlappedSpeechPenalty(gamma, beta, normalize_weights)
#         self.normalize = EmbeddingNormalization(norm)

#     @staticmethod
#     def from_pretrained(
#         model,
#         gamma: float = 3,
#         beta: float = 10,
#         norm: Union[float, torch.Tensor] = 1,
#         use_hf_token: Union[Text, bool, None] = True,
#         normalize_weights: bool = False,
#         device: Optional[torch.device] = None,
#     ):
#         model = EmbeddingModel.from_pretrained(model, use_hf_token)
#         return SpkrNetSpeakerEmbedding(
#             model, gamma, beta, norm, normalize_weights, device
#         )
    
#     def __call__(
#         self, waveform: TemporalFeatures, segmentation: TemporalFeatures
#     ) -> torch.Tensor:
#         return self.normalize(self.embedding(waveform, self.osp(segmentation)))




"""ESPNET implementation """
# from typing import Optional, Union, Text

# import torch
# from einops import rearrange

# from .. import functional as F
# from ..features import TemporalFeatures, TemporalFeatureFormatter
# from ..models import EmbeddingModel


# class SpeakerEmbedding:
#     def __init__(self, model: EmbeddingModel, device: Optional[torch.device] = None):
#         self.model = model
#         self.model.eval()
#         self.device = device
#         if self.device is None:
#             self.device = torch.device("cpu")
#         self.model.to(self.device)
#         self.waveform_formatter = TemporalFeatureFormatter()
#         self.weights_formatter = TemporalFeatureFormatter()

#     @staticmethod
#     def from_pretrained(
#         model,
#         use_hf_token: Union[Text, bool, None] = True,
#         device: Optional[torch.device] = None,
#     ) -> "SpeakerEmbedding":
#         emb_model = EmbeddingModel.from_pretrained(model, use_hf_token)
#         return SpeakerEmbedding(emb_model, device)

#     def __call__(
#         self, waveform: TemporalFeatures, weights: Optional[TemporalFeatures] = None
#     ) -> torch.Tensor:
#         """
#         Calculate speaker embeddings of input audio.
#         If weights are given, calculate many speaker embeddings from the same waveform.

#         Parameters
#         ----------
#         waveform: TemporalFeatures, shape (samples, channels) or (batch, samples, channels)
#         weights: Optional[TemporalFeatures], shape (frames, speakers) or (batch, frames, speakers)
#             Per-speaker and per-frame weights. Defaults to no weights.

#         Returns
#         -------
#         embeddings: torch.Tensor
#             If weights are provided, the shape is (batch, speakers, embedding_dim),
#             otherwise the shape is (batch, embedding_dim).
#             If batch size == 1, the batch dimension is omitted.
#         """
#         with torch.no_grad():
#             inputs = self.waveform_formatter.cast(waveform).to(self.device)
#             inputs = rearrange(inputs, "batch sample channel -> batch channel sample")
#             if weights is not None:
#                 weights = self.weights_formatter.cast(weights).to(self.device)
#                 batch_size, _, num_speakers = weights.shape
#                 inputs = inputs.repeat(1, num_speakers, 1)
#                 weights = rearrange(weights, "batch frame spk -> (batch spk) frame")
#                 inputs = rearrange(inputs, "batch spk sample -> (batch spk) 1 sample")
#                 output = rearrange(
#                     self.model(inputs, weights),
#                     "(batch spk) feat -> batch spk feat",
#                     batch=batch_size,
#                     spk=num_speakers,
#                 )
#             else:
#                 output = self.model(inputs)
#             return output.squeeze().cpu()


# class OverlappedSpeechPenalty:
#     """Applies a penalty on overlapping speech and low-confidence regions to speaker segmentation scores.

#     .. note::
#         For more information, see `"Overlap-Aware Low-Latency Online Speaker Diarization
#         based on End-to-End Local Segmentation" <https://github.com/juanmc2005/diart/blob/main/paper.pdf>`_
#         (Section 2.2.1 Segmentation-driven speaker embedding). This block implements Equation 2.

#     Parameters
#     ----------
#     gamma: float, optional
#         Exponent to lower low-confidence predictions.
#         Defaults to 3.
#     beta: float, optional
#         Temperature parameter (actually 1/beta) to lower joint speaker activations.
#         Defaults to 10.
#     normalize: bool, optional
#         Whether to min-max normalize weights to be in the range [0, 1].
#         Defaults to False.
#     """

#     def __init__(self, gamma: float = 3, beta: float = 10, normalize: bool = False):
#         self.gamma = gamma
#         self.beta = beta
#         self.formatter = TemporalFeatureFormatter()
#         self.normalize = normalize

#     def __call__(self, segmentation: TemporalFeatures) -> TemporalFeatures:
#         weights = self.formatter.cast(segmentation)  # shape (batch, frames, speakers)
#         with torch.inference_mode():
#             weights = F.overlapped_speech_penalty(weights, self.gamma, self.beta)
#             if self.normalize:
#                 min_values = weights.min(dim=1, keepdim=True).values
#                 max_values = weights.max(dim=1, keepdim=True).values
#                 weights = (weights - min_values) / (max_values - min_values)
#                 weights.nan_to_num_(1e-8)
#         return self.formatter.restore_type(weights)


# class EmbeddingNormalization:
#     def __init__(self, norm: Union[float, torch.Tensor] = 1):
#         self.norm = norm
#         # Add batch dimension if missing
#         if isinstance(self.norm, torch.Tensor) and self.norm.ndim == 2:
#             self.norm = self.norm.unsqueeze(0)

#     def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
#         with torch.inference_mode():
#             norm_embs = F.normalize_embeddings(embeddings, self.norm)
#         return norm_embs


# class OverlapAwareSpeakerEmbedding:
#     """
#     Extract overlap-aware speaker embeddings given an audio chunk and its segmentation.

#     Parameters
#     ----------
#     model: EmbeddingModel
#         A pre-trained embedding model.
#     gamma: float, optional
#         Exponent to lower low-confidence predictions.
#         Defaults to 3.
#     beta: float, optional
#         Softmax's temperature parameter (actually 1/beta) to lower joint speaker activations.
#         Defaults to 10.
#     norm: float or torch.Tensor of shape (batch, speakers, 1) where batch is optional
#         The target norm for the embeddings. It can be different for each speaker.
#         Defaults to 1.
#     normalize_weights: bool, optional
#         Whether to min-max normalize embedding weights to be in the range [0, 1].
#     device: Optional[torch.device]
#         The device on which to run the embedding model.
#         Defaults to GPU if available or CPU if not.
#     """

#     def __init__(
#         self,
#         model: EmbeddingModel,
#         gamma: float = 3,
#         beta: float = 10,
#         norm: Union[float, torch.Tensor] = 1,
#         normalize_weights: bool = False,
#         device: Optional[torch.device] = None,
#     ):
#         self.embedding = SpeakerEmbedding(model, device)
#         self.osp = OverlappedSpeechPenalty(gamma, beta, normalize_weights)
#         self.normalize = EmbeddingNormalization(norm)

#     @staticmethod
#     def from_pretrained(
#         model,
#         gamma: float = 3,
#         beta: float = 10,
#         norm: Union[float, torch.Tensor] = 1,
#         use_hf_token: Union[Text, bool, None] = True,
#         normalize_weights: bool = False,
#         device: Optional[torch.device] = None,
#     ):
#         model = EmbeddingModel.from_pretrained(model, use_hf_token)
#         return OverlapAwareSpeakerEmbedding(
#             model, gamma, beta, norm, normalize_weights, device
#         )

#     def __call__(
#         self, waveform: TemporalFeatures, segmentation: TemporalFeatures
#     ) -> torch.Tensor:
#         return self.normalize(self.embedding(waveform, self.osp(segmentation)))



from typing import Optional, Union, Text

import torch
from einops import rearrange

from .. import functional as F
from ..features import TemporalFeatures, TemporalFeatureFormatter
from ..models import EmbeddingModel

class SpeakerEmbedding:
    def __init__(self, model: EmbeddingModel, device: Optional[torch.device] = None, espnet :  Optional[bool] = False):
        self.espnet = espnet
        if espnet:
            from espnet2.bin.spk_inference import Speech2Embedding
            # config_path_joint_wavlm_ecapa = "/home/narayana/anaconda3/envs/spkr_diarize/lib/python3.10/site-packages/espnet_model_zoo/models--espnet--voxcelebs12_ecapa_wavlm_joint/snapshots/e50c506b4adce4de2cac862fa2bfa441712c5851/exp/spk_train_ecapa_Vox12_emb192_torchmelspec_subcentertopk_wavlm_jt11_raw_sp/config.yaml"
            config_path_joint_wavlm_ecapa = "/home/narayana/Projects/spkr_diarize/pretrained_models/ecapa_wavlm/config.yaml"
            model_path_joint_wavlm_ecapa = "/home/narayana/Projects/spkr_diarize/pretrained_models/ecapa_wavlm/8epoch.pth"
            # model_path_joint_wavlm_ecapa = "/home/narayana/anaconda3/envs/spkr_diarize/lib/python3.10/site-packages/espnet_model_zoo/models--espnet--voxcelebs12_ecapa_wavlm_joint/snapshots/e50c506b4adce4de2cac862fa2bfa441712c5851/exp/spk_train_ecapa_Vox12_emb192_torchmelspec_subcentertopk_wavlm_jt11_raw_sp/8epoch.pth"
            # self.model = Speech2Embedding.from_pretrained(model_tag="espnet/voxcelebs12_ecapa_wavlm_joint", device="cuda")
            self.model = Speech2Embedding(train_config=config_path_joint_wavlm_ecapa, model_file=model_path_joint_wavlm_ecapa, device='cpu')
            self.device = device
            # print(self.model)
            if self.device is None:
                self.device = torch.device("cpu")
            self.waveform_formatter = TemporalFeatureFormatter()
            self.weights_formatter = TemporalFeatureFormatter()
        else:
            self.model = model
            self.model.eval()
            self.device = device
            if self.device is None:
                self.device = torch.device("cpu")
            self.model.to(self.device)
            self.waveform_formatter = TemporalFeatureFormatter()
            self.weights_formatter = TemporalFeatureFormatter()

    @staticmethod
    def from_pretrained(
        model,
        use_hf_token: Union[Text, bool, None] = True,
        device: Optional[torch.device] = None,
    ) -> "SpeakerEmbedding":
        emb_model = EmbeddingModel.from_pretrained(model, use_hf_token)
        return SpeakerEmbedding(emb_model, device)

    def __call__(
        self, waveform: TemporalFeatures, weights: Optional[TemporalFeatures] = None
    ) -> torch.Tensor:
        """
        Calculate speaker embeddings of input audio.
        If weights are given, calculate many speaker embeddings from the same waveform.

        Parameters
        ----------
        waveform: TemporalFeatures, shape (samples, channels) or (batch, samples, channels)
        weights: Optional[TemporalFeatures], shape (frames, speakers) or (batch, frames, speakers)
            Per-speaker and per-frame weights. Defaults to no weights.

        Returns
        -------
        embeddings: torch.Tensor
            If weights are provided, the shape is (batch, speakers, embedding_dim),
            otherwise the shape is (batch, embedding_dim).
            If batch size == 1, the batch dimension is omitted.
        """
        with torch.no_grad():
            inputs = self.waveform_formatter.cast(waveform).to(self.device)
            inputs = rearrange(inputs, "batch sample channel -> batch channel sample")
            if weights is not None and not self.espnet:
                weights = self.weights_formatter.cast(weights).to(self.device)
                batch_size, _, num_speakers = weights.shape
                inputs = inputs.repeat(1, num_speakers, 1)
                weights = rearrange(weights, "batch frame spk -> (batch spk) frame")
                inputs = rearrange(inputs, "batch spk sample -> (batch spk) 1 sample")
                output = rearrange(
                    self.model(inputs, weights), # inputs =>  3,1,80000 and weights 3,293
                    "(batch spk) feat -> batch spk feat",
                    batch=batch_size,
                    spk=num_speakers,
                )
            elif weights is not None and self.espnet==True:
                weights = self.weights_formatter.cast(weights).to(self.device)
                batch_size, _, num_speakers = weights.shape
                inputs = inputs.repeat(1, num_speakers, 1) # 3,1,80k
                weights = rearrange(weights, "batch frame spk -> (batch spk) frame")
                inputs = rearrange(inputs, "batch spk sample -> (batch spk) 1 sample") # 3,1,80000
                output = []
                inputs = inputs.squeeze(dim=1)
                for i in inputs:
                    ind_input = i.squeeze()
                    ind_output = self.model(ind_input)
                    output.append(ind_output)
                output = torch.stack(output)
                output = output.squeeze(dim=1)
                return output.cpu()
            else:
                output = self.model(inputs)
            return output.squeeze().cpu()


class OverlappedSpeechPenalty:
    """Applies a penalty on overlapping speech and low-confidence regions to speaker segmentation scores.

    .. note::
        For more information, see `"Overlap-Aware Low-Latency Online Speaker Diarization
        based on End-to-End Local Segmentation" <https://github.com/juanmc2005/diart/blob/main/paper.pdf>`_
        (Section 2.2.1 Segmentation-driven speaker embedding). This block implements Equation 2.

    Parameters
    ----------
    gamma: float, optional
        Exponent to lower low-confidence predictions.
        Defaults to 3.
    beta: float, optional
        Temperature parameter (actually 1/beta) to lower joint speaker activations.
        Defaults to 10.
    normalize: bool, optional
        Whether to min-max normalize weights to be in the range [0, 1].
        Defaults to False.
    """

    def __init__(self, gamma: float = 3, beta: float = 10, normalize: bool = False):
        self.gamma = gamma
        self.beta = beta
        self.formatter = TemporalFeatureFormatter()
        self.normalize = normalize

    def __call__(self, segmentation: TemporalFeatures) -> TemporalFeatures:
        weights = self.formatter.cast(segmentation)  # shape (batch, frames, speakers)
        with torch.inference_mode():
            weights = F.overlapped_speech_penalty(weights, self.gamma, self.beta)
            if self.normalize:
                min_values = weights.min(dim=1, keepdim=True).values
                max_values = weights.max(dim=1, keepdim=True).values
                weights = (weights - min_values) / (max_values - min_values)
                weights.nan_to_num_(1e-8)
        return self.formatter.restore_type(weights)


class EmbeddingNormalization:
    def __init__(self, norm: Union[float, torch.Tensor] = 1):
        self.norm = norm
        # Add batch dimension if missing
        if isinstance(self.norm, torch.Tensor) and self.norm.ndim == 2:
            self.norm = self.norm.unsqueeze(0)

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            norm_embs = F.normalize_embeddings(embeddings, self.norm)
        return norm_embs


class OverlapAwareSpeakerEmbedding:
    """
    Extract overlap-aware speaker embeddings given an audio chunk and its segmentation.

    Parameters
    ----------
    model: EmbeddingModel
        A pre-trained embedding model.
    gamma: float, optional
        Exponent to lower low-confidence predictions.
        Defaults to 3.
    beta: float, optional
        Softmax's temperature parameter (actually 1/beta) to lower joint speaker activations.
        Defaults to 10.
    norm: float or torch.Tensor of shape (batch, speakers, 1) where batch is optional
        The target norm for the embeddings. It can be different for each speaker.
        Defaults to 1.
    normalize_weights: bool, optional
        Whether to min-max normalize embedding weights to be in the range [0, 1].
    device: Optional[torch.device]
        The device on which to run the embedding model.
        Defaults to GPU if available or CPU if not.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        gamma: float = 3,
        beta: float = 10,
        norm: Union[float, torch.Tensor] = 1,
        normalize_weights: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.embedding = SpeakerEmbedding(model, device)
        self.osp = OverlappedSpeechPenalty(gamma, beta, normalize_weights)
        self.normalize = EmbeddingNormalization(norm)

    @staticmethod
    def from_pretrained(
        model,
        gamma: float = 3,
        beta: float = 10,
        norm: Union[float, torch.Tensor] = 1,
        use_hf_token: Union[Text, bool, None] = True,
        normalize_weights: bool = False,
        device: Optional[torch.device] = None,
    ):
        model = EmbeddingModel.from_pretrained(model, use_hf_token)
        return OverlapAwareSpeakerEmbedding(
            model, gamma, beta, norm, normalize_weights, device
        )

    def __call__(
        self, waveform: TemporalFeatures, segmentation: TemporalFeatures
    ) -> torch.Tensor:
        return self.normalize(self.embedding(waveform, self.osp(segmentation)))

class EpsnetSpeakerEmbedding:
    def __init__(
        self,
        model: EmbeddingModel,
        gamma: float = 3,
        beta: float = 10,
        norm: Union[float, torch.Tensor] = 1,
        normalize_weights: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.embedding = SpeakerEmbedding(model, device, espnet=True)
        self.osp = OverlappedSpeechPenalty(gamma, beta, normalize_weights)
        self.normalize = EmbeddingNormalization(norm)

    @staticmethod
    def from_pretrained(
        model,
        gamma: float = 3,
        beta: float = 10,
        norm: Union[float, torch.Tensor] = 1,
        use_hf_token: Union[Text, bool, None] = True,
        normalize_weights: bool = False,
        device: Optional[torch.device] = None,
    ):
        model = EmbeddingModel.from_pretrained(model, use_hf_token)
        return EpsnetSpeakerEmbedding(
            model, gamma, beta, norm, normalize_weights, device
        )
    
    def __call__(
        self, waveform: TemporalFeatures, segmentation: TemporalFeatures
    ) -> torch.Tensor:
        return self.normalize(self.embedding(waveform, self.osp(segmentation)))
