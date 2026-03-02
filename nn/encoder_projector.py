from typing import Sequence


class EncoderProjector(nn.Model):
    def __init__(
        self,
        backbone: nn.Model,
        hidden_channels: int | Sequence[int],
        head_channels: int,
        head_norm: bool,
        name: str = None,
    ):
        with nn.normalization_scope(nn.BatchNormalization), parameter_scope(nn.BatchNormalization, synchronized=True):
            projector = nn.MLP([*convert_to_list(hidden_channels), head_channels], head_bias=False, head_norm=head_norm, name="projector")

        features = backbone.outputs[-1]
        embeddings = {
            4: nn.GlobalAveragePooling2D(),
            5: nn.GlobalAveragePooling3D(),
        }[features.shape.rank](features)
        projections = projector(embeddings)

        super().__init__(
            inputs=backbone.input,
            outputs={
                "embedding": embeddings,
                "projection": projections,
            },
            name=name or backbone.name
        )
        self.backbone = backbone
        self.projector = projector
        self.ckpt_items = dict(
            backbone=self.backbone,
            projector=self.projector,
        )
