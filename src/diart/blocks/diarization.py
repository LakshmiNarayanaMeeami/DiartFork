# from __future__ import annotations

# from typing import Sequence

# import numpy as np
# import torch
# from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow, Segment
# from pyannote.metrics.base import BaseMetric
# from pyannote.metrics.diarization import DiarizationErrorRate
# from typing_extensions import Literal

# from . import base
# from .aggregation import DelayedAggregation
# from .clustering import OnlineSpeakerClustering
# from .embedding import OverlapAwareSpeakerEmbedding, SpkrNetSpeakerEmbedding
# from .segmentation import SpeakerSegmentation
# from .utils import Binarize
# from .. import models as m


# class SpeakerDiarizationConfig(base.PipelineConfig):
#     def __init__(
#         self,
#         segmentation: m.SegmentationModel | None = None,
#         embedding: m.EmbeddingModel | None = None,
#         duration: float = 5,
#         step: float = 0.5,
#         latency: float | Literal["max", "min"] | None = None,
#         tau_active: float = 0.6,
#         rho_update: float = 0.3,
#         delta_new: float = 1,
#         gamma: float = 3,
#         beta: float = 10,
#         max_speakers: int = 20,
#         normalize_embedding_weights: bool = False,
#         device: torch.device | None = None,
#         sample_rate: int = 16000,
#         **kwargs,
#     ):
#         # Default segmentation model is pyannote/segmentation
#         self.segmentation = segmentation or m.SegmentationModel.from_pyannote(
#             "pyannote/segmentation"
#         )

#         # Default embedding model is pyannote/embedding
#         self.embedding = embedding or m.EmbeddingModel.from_pyannote(
#             "pyannote/embedding"
#         )

#         self._duration = duration
#         self._sample_rate = sample_rate

#         # Latency defaults to the step duration
#         self._step = step
#         self._latency = latency
#         if self._latency is None or self._latency == "min":
#             self._latency = self._step
#         elif self._latency == "max":
#             self._latency = self._duration

#         self.tau_active = tau_active
#         self.rho_update = rho_update
#         self.delta_new = delta_new
#         self.gamma = gamma
#         self.beta = beta
#         self.max_speakers = max_speakers
#         self.normalize_embedding_weights = normalize_embedding_weights
#         self.device = device or torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )

#     @property
#     def duration(self) -> float:
#         return self._duration

#     @property
#     def step(self) -> float:
#         return self._step

#     @property
#     def latency(self) -> float:
#         return self._latency

#     @property
#     def sample_rate(self) -> int:
#         return self._sample_rate


# class SpeakerDiarization(base.Pipeline):
#     def __init__(self, config: SpeakerDiarizationConfig | None = None, emb_model = None):
#         self._config = SpeakerDiarizationConfig() if config is None else config

#         msg = f"Latency should be in the range [{self._config.step}, {self._config.duration}]"
#         assert self._config.step <= self._config.latency <= self._config.duration, msg

#         self.segmentation = SpeakerSegmentation(
#             self._config.segmentation, self._config.device
#         )

#         # if emb_model == "espnet":
#         #     self.embedding = EpsnetSpeakerEmbedding(
#         #         self._config.embedding,
#         #         self._config.gamma,
#         #         self._config.beta,
#         #         norm=1,
#         #         normalize_weights=self._config.normalize_embedding_weights,
#         #         device=self._config.device,
#         #     )
#         if emb_model == "spkrnet":
#             self.embedding = SpkrNetSpeakerEmbedding(
#                 self._config.embedding,
#                 self._config.gamma,
#                 self._config.beta,
#                 norm=1,
#                 normalize_weights=self._config.normalize_embedding_weights,
#                 device=self._config.device,
#             )
#         else:
#             self.embedding = OverlapAwareSpeakerEmbedding(
#                 self._config.embedding,
#                 self._config.gamma,
#                 self._config.beta,
#                 norm=1,
#                 normalize_weights=self._config.normalize_embedding_weights,
#                 device=self._config.device,
#             )
#         self.pred_aggregation = DelayedAggregation(
#             self._config.step,
#             self._config.latency,
#             strategy="hamming",
#             cropping_mode="loose",
#         )
#         self.audio_aggregation = DelayedAggregation(
#             self._config.step,
#             self._config.latency,
#             strategy="first",
#             cropping_mode="center",
#         )
#         self.binarize = Binarize(self._config.tau_active)

#         # Internal state, handle with care
#         self.timestamp_shift = 0
#         self.clustering = None
#         self.chunk_buffer, self.pred_buffer = [], []
#         self.reset()

#     @staticmethod
#     def get_config_class() -> type:
#         return SpeakerDiarizationConfig

#     @staticmethod
#     def suggest_metric() -> BaseMetric:
#         return DiarizationErrorRate(collar=0, skip_overlap=False)

#     @staticmethod
#     def hyper_parameters() -> Sequence[base.HyperParameter]:
#         return [base.TauActive, base.RhoUpdate, base.DeltaNew]

#     @property
#     def config(self) -> SpeakerDiarizationConfig:
#         return self._config

#     def set_timestamp_shift(self, shift: float):
#         self.timestamp_shift = shift

#     def reset(self):
#         self.set_timestamp_shift(0)
#         self.clustering = OnlineSpeakerClustering(
#             self.config.tau_active,
#             self.config.rho_update,
#             self.config.delta_new,
#             "cosine",
#             self.config.max_speakers,
#         )
#         self.chunk_buffer, self.pred_buffer = [], []

#     def __call__(
#         self, waveforms: Sequence[SlidingWindowFeature]
#     ) -> Sequence[tuple[Annotation, SlidingWindowFeature]]:
#         """Diarize the next audio chunks of an audio stream.

#         Parameters
#         ----------
#         waveforms: Sequence[SlidingWindowFeature]
#             A sequence of consecutive audio chunks from an audio stream.

#         Returns
#         -------
#         Sequence[tuple[Annotation, SlidingWindowFeature]]
#             Speaker diarization of each chunk alongside their corresponding audio.
#         """
#         batch_size = len(waveforms)
#         msg = "Pipeline expected at least 1 input"
#         assert batch_size >= 1, msg

#         # Create batch from chunk sequence, shape (batch, samples, channels)
#         batch = torch.stack([torch.from_numpy(w.data) for w in waveforms])

#         expected_num_samples = int(
#             np.rint(self.config.duration * self.config.sample_rate)
#         )
#         msg = f"Expected {expected_num_samples} samples per chunk, but got {batch.shape[1]}"
#         assert batch.shape[1] == expected_num_samples, msg

#         # Extract segmentation and embeddings
#         segmentations = self.segmentation(batch)  # shape (batch, frames, speakers)
#         # embeddings has shape (batch, speakers, emb_dim)
#         embeddings = self.embedding(batch, segmentations)

#         seg_resolution = waveforms[0].extent.duration / segmentations.shape[1]

#         outputs = []
#         for wav, seg, emb in zip(waveforms, segmentations, embeddings):
#             # Add timestamps to segmentation
#             sw = SlidingWindow(
#                 start=wav.extent.start,
#                 duration=seg_resolution,
#                 step=seg_resolution,
#             )
#             seg = SlidingWindowFeature(seg.cpu().numpy(), sw)

#             # Update clustering state and permute segmentation
#             permuted_seg = self.clustering(seg, emb)

#             # Update sliding buffer
#             self.chunk_buffer.append(wav)
#             self.pred_buffer.append(permuted_seg)

#             # Aggregate buffer outputs for this time step
#             agg_waveform = self.audio_aggregation(self.chunk_buffer)
#             agg_prediction = self.pred_aggregation(self.pred_buffer)
#             agg_prediction = self.binarize(agg_prediction)

#             # Shift prediction timestamps if required
#             if self.timestamp_shift != 0:
#                 shifted_agg_prediction = Annotation(agg_prediction.uri)
#                 for segment, track, speaker in agg_prediction.itertracks(
#                     yield_label=True
#                 ):
#                     new_segment = Segment(
#                         segment.start + self.timestamp_shift,
#                         segment.end + self.timestamp_shift,
#                     )
#                     shifted_agg_prediction[new_segment, track] = speaker
#                 agg_prediction = shifted_agg_prediction

#             outputs.append((agg_prediction, agg_waveform))

#             # Make place for new chunks in buffer if required
#             if len(self.chunk_buffer) == self.pred_aggregation.num_overlapping_windows:
#                 self.chunk_buffer = self.chunk_buffer[1:]
#                 self.pred_buffer = self.pred_buffer[1:]

#         return outputs


""" Diarixation for espnet"""

# from __future__ import annotations

# from typing import Sequence

# import numpy as np
# import torch
# from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow, Segment
# from pyannote.metrics.base import BaseMetric
# from pyannote.metrics.diarization import DiarizationErrorRate
# from typing_extensions import Literal

# from . import base
# from .aggregation import DelayedAggregation
# from .clustering import OnlineSpeakerClustering
# from .embedding import OverlapAwareSpeakerEmbedding
# from .segmentation import SpeakerSegmentation
# from .utils import Binarize
# from .. import models as m


# class SpeakerDiarizationConfig(base.PipelineConfig):
#     def __init__(
#         self,
#         segmentation: m.SegmentationModel | None = None,
#         embedding: m.EmbeddingModel | None = None,
#         duration: float = 5,
#         step: float = 0.5,
#         latency: float | Literal["max", "min"] | None = None,
#         tau_active: float = 0.6,
#         rho_update: float = 0.3,
#         delta_new: float = 1,
#         gamma: float = 3,
#         beta: float = 10,
#         max_speakers: int = 20,
#         normalize_embedding_weights: bool = False,
#         device: torch.device | None = None,
#         sample_rate: int = 16000,
#         **kwargs,
#     ):
#         # Default segmentation model is pyannote/segmentation
#         self.segmentation = segmentation or m.SegmentationModel.from_pyannote(
#             "pyannote/segmentation"
#         )

#         # Default embedding model is pyannote/embedding
#         self.embedding = embedding or m.EmbeddingModel.from_pyannote(
#             "pyannote/embedding"
#         )

#         self._duration = duration
#         self._sample_rate = sample_rate

#         # Latency defaults to the step duration
#         self._step = step
#         self._latency = latency
#         if self._latency is None or self._latency == "min":
#             self._latency = self._step
#         elif self._latency == "max":
#             self._latency = self._duration

#         self.tau_active = tau_active
#         self.rho_update = rho_update
#         self.delta_new = delta_new
#         self.gamma = gamma
#         self.beta = beta
#         self.max_speakers = max_speakers
#         self.normalize_embedding_weights = normalize_embedding_weights
#         self.device = device or torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )

#     @property
#     def duration(self) -> float:
#         return self._duration

#     @property
#     def step(self) -> float:
#         return self._step

#     @property
#     def latency(self) -> float:
#         return self._latency

#     @property
#     def sample_rate(self) -> int:
#         return self._sample_rate


# class SpeakerDiarization(base.Pipeline):
#     def __init__(self, config: SpeakerDiarizationConfig | None = None):
#         self._config = SpeakerDiarizationConfig() if config is None else config

#         msg = f"Latency should be in the range [{self._config.step}, {self._config.duration}]"
#         assert self._config.step <= self._config.latency <= self._config.duration, msg

#         self.segmentation = SpeakerSegmentation(
#             self._config.segmentation, self._config.device
#         )
#         self.embedding = OverlapAwareSpeakerEmbedding(
#             self._config.embedding,
#             self._config.gamma,
#             self._config.beta,
#             norm=1,
#             normalize_weights=self._config.normalize_embedding_weights,
#             device=self._config.device,
#         )
#         self.pred_aggregation = DelayedAggregation(
#             self._config.step,
#             self._config.latency,
#             strategy="hamming",
#             cropping_mode="loose",
#         )
#         self.audio_aggregation = DelayedAggregation(
#             self._config.step,
#             self._config.latency,
#             strategy="first",
#             cropping_mode="center",
#         )
#         self.binarize = Binarize(self._config.tau_active)

#         # Internal state, handle with care
#         self.timestamp_shift = 0
#         self.clustering = None
#         self.chunk_buffer, self.pred_buffer = [], []
#         self.reset()

#     @staticmethod
#     def get_config_class() -> type:
#         return SpeakerDiarizationConfig

#     @staticmethod
#     def suggest_metric() -> BaseMetric:
#         return DiarizationErrorRate(collar=0, skip_overlap=False)

#     @staticmethod
#     def hyper_parameters() -> Sequence[base.HyperParameter]:
#         return [base.TauActive, base.RhoUpdate, base.DeltaNew]

#     @property
#     def config(self) -> SpeakerDiarizationConfig:
#         return self._config

#     def set_timestamp_shift(self, shift: float):
#         self.timestamp_shift = shift

#     def reset(self):
#         self.set_timestamp_shift(0)
#         self.clustering = OnlineSpeakerClustering(
#             self.config.tau_active,
#             self.config.rho_update,
#             self.config.delta_new,
#             "cosine",
#             self.config.max_speakers,
#         )
#         self.chunk_buffer, self.pred_buffer = [], []

#     def __call__(
#         self, waveforms: Sequence[SlidingWindowFeature]
#     ) -> Sequence[tuple[Annotation, SlidingWindowFeature]]:
#         """Diarize the next audio chunks of an audio stream.

#         Parameters
#         ----------
#         waveforms: Sequence[SlidingWindowFeature]
#             A sequence of consecutive audio chunks from an audio stream.

#         Returns
#         -------
#         Sequence[tuple[Annotation, SlidingWindowFeature]]
#             Speaker diarization of each chunk alongside their corresponding audio.
#         """
#         batch_size = len(waveforms)
#         msg = "Pipeline expected at least 1 input"
#         assert batch_size >= 1, msg

#         # Create batch from chunk sequence, shape (batch, samples, channels)
#         batch = torch.stack([torch.from_numpy(w.data) for w in waveforms])

#         expected_num_samples = int(
#             np.rint(self.config.duration * self.config.sample_rate)
#         )
#         msg = f"Expected {expected_num_samples} samples per chunk, but got {batch.shape[1]}"
#         assert batch.shape[1] == expected_num_samples, msg

#         # Extract segmentation and embeddings
#         segmentations = self.segmentation(batch)  # shape (batch, frames, speakers)
#         # embeddings has shape (batch, speakers, emb_dim)
#         embeddings = self.embedding(batch, segmentations)

#         seg_resolution = waveforms[0].extent.duration / segmentations.shape[1]

#         outputs = []
#         for wav, seg, emb in zip(waveforms, segmentations, embeddings):
#             # Add timestamps to segmentation
#             sw = SlidingWindow(
#                 start=wav.extent.start,
#                 duration=seg_resolution,
#                 step=seg_resolution,
#             )
#             seg = SlidingWindowFeature(seg.cpu().numpy(), sw)

#             # Update clustering state and permute segmentation
#             permuted_seg = self.clustering(seg, emb)

#             # Update sliding buffer
#             self.chunk_buffer.append(wav)
#             self.pred_buffer.append(permuted_seg)

#             # Aggregate buffer outputs for this time step
#             agg_waveform = self.audio_aggregation(self.chunk_buffer)
#             agg_prediction = self.pred_aggregation(self.pred_buffer)
#             agg_prediction = self.binarize(agg_prediction)

#             # Shift prediction timestamps if required
#             if self.timestamp_shift != 0:
#                 shifted_agg_prediction = Annotation(agg_prediction.uri)
#                 for segment, track, speaker in agg_prediction.itertracks(
#                     yield_label=True
#                 ):
#                     new_segment = Segment(
#                         segment.start + self.timestamp_shift,
#                         segment.end + self.timestamp_shift,
#                     )
#                     shifted_agg_prediction[new_segment, track] = speaker
#                 agg_prediction = shifted_agg_prediction

#             outputs.append((agg_prediction, agg_waveform))

#             # Make place for new chunks in buffer if required
#             if len(self.chunk_buffer) == self.pred_aggregation.num_overlapping_windows:
#                 self.chunk_buffer = self.chunk_buffer[1:]
#                 self.pred_buffer = self.pred_buffer[1:]

#         return outputs
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow, Segment
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.diarization import DiarizationErrorRate
from typing_extensions import Literal

from . import base
from .aggregation import DelayedAggregation
from .clustering import OnlineSpeakerClustering
from .embedding import OverlapAwareSpeakerEmbedding, EpsnetSpeakerEmbedding
from .segmentation import SpeakerSegmentation
from .utils import Binarize
from .. import models as m

from scipy.spatial.distance import cosine

class SpeakerDiarizationConfig(base.PipelineConfig):
    def __init__(
        self,
        segmentation: m.SegmentationModel | None = None,
        embedding: m.EmbeddingModel | None = None,
        duration: float = 5,
        step: float = 0.5,
        latency: float | Literal["max", "min"] | None = None,
        tau_active: float = 0.6,
        rho_update: float = 0.3,
        delta_new: float = 1,
        gamma: float = 3,
        beta: float = 10,
        max_speakers: int = 20,
        normalize_embedding_weights: bool = False,
        device: torch.device | None = None,
        sample_rate: int = 16000,
        **kwargs,
    ):
        # Default segmentation model is pyannote/segmentation
        self.segmentation = segmentation or m.SegmentationModel.from_pyannote(
            "pyannote/segmentation"
        )

        # Default embedding model is pyannote/embedding
        self.embedding = embedding or m.EmbeddingModel.from_pyannote(
            "pyannote/embedding"
        )

        self._duration = duration
        self._sample_rate = sample_rate

        # Latency defaults to the step duration
        self._step = step
        self._latency = latency
        if self._latency is None or self._latency == "min":
            self._latency = self._step
        elif self._latency == "max":
            self._latency = self._duration

        self.tau_active = tau_active
        self.rho_update = rho_update
        self.delta_new = delta_new
        self.gamma = gamma
        self.beta = beta
        self.max_speakers = max_speakers
        self.normalize_embedding_weights = normalize_embedding_weights
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def step(self) -> float:
        return self._step

    @property
    def latency(self) -> float:
        return self._latency

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


class SpeakerDiarization(base.Pipeline):
    def __init__(self, config: SpeakerDiarizationConfig | None = None, espnet = False, known_spkr_wavs = None, centroids_update = True):
        self.centroids_update = centroids_update
        self.known_spkr_wavs_folder = known_spkr_wavs
        if self.known_spkr_wavs_folder is not None:
            import os
            self.known_spkr_wavs = {}
            from pyannote.audio import Model
            model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token="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")
            from pyannote.audio import Inference
            # DF the uri's and abs path
            inference = Inference(model, window="whole")
            abs_folder_path = os.path.abspath(self.known_spkr_wavs_folder)
            for file in os.listdir(abs_folder_path):
                if file.endswith(".wav"):
                    abs_path = os.path.join(abs_folder_path, file)
                    print("Loaded Audio : ", file)
                    uri = str(file).split('.')[0]
                    self.known_spkr_wavs[uri] = abs_path
            self.known_spkr_embds = {}
            # extract the embeddings
            for name, file_path in self.known_spkr_wavs.items():
                self.known_spkr_embds[name] = inference(file_path)

            self.referenced_embds = np.concatenate([np.expand_dims(i, axis=0) for i in self.known_spkr_embds.values()], axis=0)
            self.names_order = [i for i in self.known_spkr_embds.keys()]
            self.speaker_identity_map = {f'speaker{idx}':name for idx, name in enumerate(self.names_order)}

        self._config = SpeakerDiarizationConfig() if config is None else config

        msg = f"Latency should be in the range [{self._config.step}, {self._config.duration}]"
        assert self._config.step <= self._config.latency <= self._config.duration, msg

        self.segmentation = SpeakerSegmentation(
            self._config.segmentation, self._config.device
        )
        if espnet:
            self.embedding = EpsnetSpeakerEmbedding(
                self._config.embedding,
                self._config.gamma,
                self._config.beta,
                norm=1,
                normalize_weights=self._config.normalize_embedding_weights,
                device=self._config.device,
            )
        else:
            self.embedding = OverlapAwareSpeakerEmbedding(
                self._config.embedding,
                self._config.gamma,
                self._config.beta,
                norm=1,
                normalize_weights=self._config.normalize_embedding_weights,
                device=self._config.device,
            )
        self.pred_aggregation = DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="hamming",
            cropping_mode="loose",
        )
        self.audio_aggregation = DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="first",
            cropping_mode="center",
        )
        self.binarize = Binarize(self._config.tau_active)

        # Internal state, handle with care
        self.timestamp_shift = 0
        self.clustering = None
        self.chunk_buffer, self.pred_buffer = [], []
        self.reset()

    @staticmethod
    def get_config_class() -> type:
        return SpeakerDiarizationConfig

    @staticmethod
    def suggest_metric() -> BaseMetric:
        return DiarizationErrorRate(collar=0, skip_overlap=False)

    @staticmethod
    def hyper_parameters() -> Sequence[base.HyperParameter]:
        return [base.TauActive, base.RhoUpdate, base.DeltaNew]

    @property
    def config(self) -> SpeakerDiarizationConfig:
        return self._config

    def set_timestamp_shift(self, shift: float):
        self.timestamp_shift = shift

    def reset(self):
        self.set_timestamp_shift(0)
        if self.known_spkr_wavs_folder is not None and self.centroids_update is True:
            self.clustering = OnlineSpeakerClustering(
                self.config.tau_active,
                self.config.rho_update,
                self.config.delta_new,
                "cosine",
                self.config.max_speakers,
                referenced_embds = self.referenced_embds
            )
        else:
            self.clustering = OnlineSpeakerClustering(
                self.config.tau_active,
                self.config.rho_update,
                self.config.delta_new,
                "cosine",
                self.config.max_speakers,
                referenced_embds = None
            )
        self.chunk_buffer, self.pred_buffer = [], []

    def __call__(
        self, waveforms: Sequence[SlidingWindowFeature]
    ) -> Sequence[tuple[Annotation, SlidingWindowFeature]]:
        """Diarize the next audio chunks of an audio stream.

        Parameters
        ----------
        waveforms: Sequence[SlidingWindowFeature]
            A sequence of consecutive audio chunks from an audio stream.

        Returns
        -------
        Sequence[tuple[Annotation, SlidingWindowFeature]]
            Speaker diarization of each chunk alongside their corresponding audio.
        """
        batch_size = len(waveforms)
        msg = "Pipeline expected at least 1 input"
        assert batch_size >= 1, msg

        # Create batch from chunk sequence, shape (batch, samples, channels)
        batch = torch.stack([torch.from_numpy(w.data) for w in waveforms])

        expected_num_samples = int(
            np.rint(self.config.duration * self.config.sample_rate)
        )
        msg = f"Expected {expected_num_samples} samples per chunk, but got {batch.shape[1]}"
        assert batch.shape[1] == expected_num_samples, msg

        # Extract segmentation and embeddings
        segmentations = self.segmentation(batch)  # shape (batch, frames, speakers)
        # embeddings has shape (batch, speakers, emb_dim)
        embeddings = self.embedding(batch, segmentations)
        """
        Edit: Feb 04 2024
            Preparing emebddings from the wavlm features by summing over the time axis with constratin fo one emebdding per speaker
            Step. 1: Getting the features of the wavlm along with the segmentations (activity probs.)
            Ex: segmentations, wavlm_features = self.segmentation(batch)
            Step. 2: Binarize the segmentations to get the speaker activity
            Ex: spkr_activity = self.binarize(segmentations<should be sliding window objecet>)
            instead directly apply brute force binarization
            Step. 3: Get the speaker based-embeddings from the wavlm features
            Ex: embeddings = np.sum(wavlm_features * spkr_activity, axis=1)
        """
        # segmentations, sincnet_features = self.segmentation(batch) #(batch, frames, speakers), 
        # spkr_activity = (segmentations>self._config.tau_active).int() # binarize the segmentations (batch, frames, dim)
        # we should get a single vector from the segmentation 
        # mask feature frames with no activity and with overlapping speakers
        # mask the frames with only one speaker exists.
        # mask = torch.sum(spkr_activity, dim=-1) == 1 #(1, frames)
        # # mask = mask.int()
        # print("Mask: ", mask.shape)
        
        # masked_spkr_activity = spkr_activity[:, mask.squeeze(0)]  
        
        # # select the corresponding wavlm features
        # batch_size = sincnet_features.shape[0]
        # feature_dim = sincnet_features.shape[-1]
        # embeddings = torch.zeros((batch_size, spkr_activity.shape[-1], feature_dim))
        # sincnet_features = sincnet_features.cpu()
        # masked_sincnet_features = sincnet_features[:, mask.squeeze(0), :] # (batch, active_frames, dim)
        # features = masked_sincnet_features.squeeze(0)
        # activity = masked_spkr_activity.squeeze(0) # (active_frames, num_speakers)
        # weighted_features = features * activity
        # embedding = torch.sum(weighted_features, dim=0).unsqueeze(0)
        # embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        """again Edited by me"""
        # mask = torch.sum(spkr_activity, dim=-1) == 1  # shape: (1, frames)
        # print("Mask: ", mask.shape)

        # # Apply mask to speaker activity: (1, active_frames, num_speakers)
        # masked_spkr_activity = spkr_activity[:, mask.squeeze(0), :]

        # # Select corresponding sincnet (or wavlm) features: (batch, active_frames, feature_dim)
        # sincnet_features = sincnet_features.cpu()
        # masked_sincnet_features = sincnet_features[:, mask.squeeze(0), :]

        # # For simplicity, assuming batch size = 1; remove the batch dimension
        # features = masked_sincnet_features.squeeze(0)      # (active_frames, feature_dim)
        # activity = masked_spkr_activity.squeeze(0)           # (active_frames, num_speakers)

        # # To average per speaker:
        # # Expand dimensions to allow broadcasting:
        # # features: (active_frames, 1, feature_dim)
        # # activity:  (active_frames, num_speakers, 1)
        # features_expanded = features.unsqueeze(1)   # (active_frames, 1, feature_dim)
        # activity_expanded  = activity.unsqueeze(-1)   # (active_frames, num_speakers, 1)

        # # Weight features by speaker activity. This results in a tensor of shape (active_frames, num_speakers, feature_dim)
        # weighted_features = features_expanded * activity_expanded

        # # Sum over frames for each speaker
        # speaker_sum = weighted_features.sum(dim=0)  # (num_speakers, feature_dim)

        # # Count number of frames each speaker is active
        # speaker_count = activity.sum(dim=0).unsqueeze(-1)  # (num_speakers, 1)

        # # Compute the average feature vector for each speaker.
        # # (If a speaker was never active, you may need additional handling to avoid division by zero.)
        # speaker_avg = speaker_sum / speaker_count  # (num_speakers, feature_dim)

        # # Optionally, normalize the embeddings
        # speaker_avg = torch.nn.functional.normalize(speaker_avg, p=2, dim=-1)

        # # If you need to keep the batch dimension, you can unsqueeze:
        # embeddings = speaker_avg.unsqueeze(0)  # (1, num_speakers, feature_dim)

        
        # # embeddings = torch.sum(spkr_activity.T*masked_sincnet_features.squeeze(0), dim=1).unsqueeze(0)
        # print(embeddings.shape)
        """End of edit"""
        
        seg_resolution = waveforms[0].extent.duration / segmentations.shape[1]
        outputs = []
        for wav, seg, emb in zip(waveforms, segmentations, embeddings):
            # Add timestamps to segmentation
            sw = SlidingWindow(
                start=wav.extent.start,
                duration=seg_resolution,
                step=seg_resolution,
            )
            seg = SlidingWindowFeature(seg.cpu().numpy(), sw)

            # Update clustering state and permute segmentation
            permuted_seg = self.clustering(seg, emb)

            # Update sliding buffer
            self.chunk_buffer.append(wav)
            self.pred_buffer.append(permuted_seg)

            # Aggregate buffer outputs for this time step
            agg_waveform = self.audio_aggregation(self.chunk_buffer)
            agg_prediction = self.pred_aggregation(self.pred_buffer)
            clustered_op = agg_prediction.data.copy()
            agg_prediction = self.binarize(agg_prediction)
            # print(self.speaker_identity_map)
            if self.centroids_update is True and self.known_spkr_wavs_folder is not None:
                agg_prediction = agg_prediction.rename_labels(self.speaker_identity_map)
            else:
            # Label assignment 
                if self.known_spkr_wavs_folder is not None:
                    cnt_labels = agg_prediction.labels()
                    if len(cnt_labels) != 0 and agg_waveform.data.shape[0] != 8000:
                        agg_prediction = self.match_and_identify_speakers(embeddings=emb, original_activity=seg.data, clustered_output=clustered_op, known_speaker_embeddings= self.known_spkr_embds, annotation=agg_prediction)
                    elif len(cnt_labels) != 0 and agg_waveform.data.shape[0] == 8000:
                        agg_prediction = self.match_and_identify_speakers(embeddings=emb, original_activity=seg.data[293-clustered_op.shape[0]:,:], clustered_output=clustered_op, known_speaker_embeddings= self.known_spkr_embds, annotation=agg_prediction)

            # Shift prediction timestamps if required
            if self.timestamp_shift != 0:
                shifted_agg_prediction = Annotation(agg_prediction.uri)
                for segment, track, speaker in agg_prediction.itertracks(
                    yield_label=True
                ):
                    new_segment = Segment(
                        segment.start + self.timestamp_shift,
                        segment.end + self.timestamp_shift,
                    )
                    shifted_agg_prediction[new_segment, track] = speaker
                agg_prediction = shifted_agg_prediction

            outputs.append((agg_prediction, agg_waveform))

            # Make place for new chunks in buffer if required
            if len(self.chunk_buffer) == self.pred_aggregation.num_overlapping_windows:
                self.chunk_buffer = self.chunk_buffer[1:]
                self.pred_buffer = self.pred_buffer[1:]

        return outputs
    
    # def labelling(self, windows_seg, seg_after_clustering, embeddings):
    #     seg_after_clustering = seg_after_clustering.data > self.threshold
    #     num_frames, num_embeddings = windows_seg.shape
    #     _, num_output_speakers = seg_after_clustering.shape

    #     # Find which output speakers are actually active
    #     active_output_speakers = []
    #     for spk in range(num_output_speakers):
    #         if np.any(seg_after_clustering[:, spk]):
    #             active_output_speakers.append(spk)

    #     # For each active output speaker, compute correlation with original activity patterns
    #     similarity_matrix = np.zeros((len(active_output_speakers), num_embeddings))

    #     for i, output_spk in enumerate(active_output_speakers):
    #         output_pattern = seg_after_clustering[:, output_spk]

    #         for j in range(num_embeddings):
    #             # Compute correlation between this output speaker's activity pattern
    #             # and the original activity pattern for embedding j
    #             activity_pattern_j = windows_seg[:, j]

    #             # Calculate correlation
    #             if np.std(output_pattern) > 0 and np.std(activity_pattern_j) > 0:
    #                 corr = np.corrcoef(output_pattern, activity_pattern_j)[0, 1]
    #                 similarity_matrix[i, j] = corr

    #     # Use Hungarian algorithm for optimal assignment
    #     # Note: We may have more output speakers than embeddings
    #     row_ind, col_ind = linear_sum_assignment(-similarity_matrix[:, :num_embeddings])

    #     # Create mapping from speaker label to embedding index
    #     speaker_to_embedding = {}
    #     for i, row in enumerate(row_ind):
    #         output_spk = active_output_speakers[row]
    #         embedding_idx = col_ind[i]
    #         speaker_to_embedding[f"speaker{output_spk}"] = embedding_idx

    #     return speaker_to_embedding

    def match_and_identify_speakers(self,
        embeddings: np.ndarray,             # Shape (num_embeddings, embedding_dim) (3, 512)
        original_activity: np.ndarray,      # Shape (num_frames, num_embeddings) (293, 3)
        clustered_output: np.ndarray,       # Shape (num_frames, max_speakers) e.g., (293, 20) Global
        known_speaker_embeddings: dict,     # Dict mapping speaker names to embeddings -> values: ndarray
        annotation: object = None           # Optional annotation object to update the label names
            ):
        
        binarized_output = clustered_output > self._config.tau_active

        num_frames, num_embeddings = original_activity.shape
        _, num_output_speakers = binarized_output.shape


        num_orig_frames = original_activity.shape[0]
        num_cluster_frames = clustered_output.shape[0]


        if num_orig_frames > num_cluster_frames:
            pad_size = num_orig_frames - num_cluster_frames
            clustered_output = np.pad(
                clustered_output, 
                ((0, pad_size), (0, 0)), 
                'constant', 
                constant_values=0
            )
        else:
            # Truncate clustered_output
            clustered_output = clustered_output[:num_orig_frames, :]
        
        # Step 1: Find active speakers in the output
        active_speakers = []
        for spk in range(num_output_speakers):
            if np.any(binarized_output[:, spk]):
                active_speakers.append(spk)

        # Step 2: Match embedding vectors to output speaker labels
        embedding_to_speaker_map = {} 
        speaker_to_embedding_map = {}

        # Calculate correlation matrix between cluster-output patterns and original activity
        similarity_matrix = np.zeros((len(active_speakers), num_embeddings)) # 1,3

        for i, output_spk in enumerate(active_speakers): # shape ; frames, 20
            output_pattern = binarized_output[:, output_spk]

            for j in range(num_embeddings):
                # Calculate correlation  coeef. between speaker patterns
                if np.std(output_pattern) > 0 and np.std(original_activity[:, j]) > 0:
                    corr = np.corrcoef(output_pattern, original_activity[:, j])[0, 1]
                    similarity_matrix[i, j] = corr if not np.isnan(corr) else 0

        # For each speaker, find the embedding with highest correlation
        for i, output_spk in enumerate(active_speakers):
            # TODO: Handle the embeddings such that the only one predicted embedding should be mapped to the known speakers in the window.
            # Select 2nd best
            
            best_embedding = np.argmax(similarity_matrix[i])
            # Handling : max one embedding per label in the window
            similarity_matrix[i] = None 

            # if best_embedding not in seen_embeds:
            #     best_embedding = np.argmax(similarity_matrix[i])
            #     similarity_matrix[i] = -1
                        
            embedding_to_speaker_map[best_embedding] = f"speaker{output_spk}"
            speaker_to_embedding_map[f"speaker{output_spk}"] = best_embedding

        # Step 3: Match detected speakers to known speakers
        speaker_identity_map = {}
        seen_names = []
        for speaker_label, embedding_idx in speaker_to_embedding_map.items():
            detected_embedding = embeddings[embedding_idx]

            # Calculate similarity to each known speaker
            best_match = None
            best_similarity = -1

            
            for name, known_embedding in known_speaker_embeddings.items():
                # Calculate cosine similarity (1 = identical, -1 = opposite)
                similarity = 1 - cosine(detected_embedding, known_embedding) # 1-consine_distance = cosine_simialrity

                if similarity > best_similarity and name not in seen_names:
                    best_similarity = similarity
                    best_match = name
                    seen_names.append(name)

            # Map the original speaker label to the identified name
            speaker_identity_map[speaker_label] = best_match
            # print(speaker_identity_map)
        # Step 4: Update annotation if provided
        if annotation is not None:
            # for segment, track, label in list(annotation.itertracks(yield_label=True)):
            #     if label in speaker_identity_map:
                    # Replace generic label with identified speaker name
            # tracks = annotation.get_tracks(self, Segment)
            annotation = annotation.rename_labels(speaker_identity_map)
            

        return annotation