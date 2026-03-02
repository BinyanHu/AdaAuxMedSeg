import tensorflow as tf


class RKBNet(nn.Model):
    def __init__(
        self,
        backbone: nn.Model,
        num_perms: int,
        flip: int,
        mask: bool,
        name=None
    ):
        x = backbone.outputs[-1]  # ((N8)HWC)
        ndim = x.shape.rank - 2
        if ndim == 2:
            B, H, W, C = x.shape
        else:
            B, D, H, W, C = x.shape
        n_patches_per_image = 2 ** ndim
        n_images = B // n_patches_per_image

        with parameter_scope(
            nn.Dense, kernel_initializer=nn.DENSE_KERNEL_INITIALIZER
        ):
            shared_head = []
            shared_head.append({
                1: nn.GlobalAveragePooling1D(),
                2: nn.GlobalAveragePooling2D(),
                3: nn.GlobalAveragePooling3D(),
            }[ndim])
            shared_head.append(
                nn.Dense(64),  # fc6
            )
            # ((N8),64)

            for module in shared_head:
                x = module(x)
            # NOTE reshape and concat
            x = tf.reshape(x, [n_images, n_patches_per_image * 64])  # (N,8*64)

            shared_x = x
            del x
            task_heads = []
            output_xs = {}
            for subtask in ['perm', 'flip', 'mask']:
                if subtask == 'perm':
                    head = [
                        nn.Dense(1024),
                        nn.ReLU(),
                        nn.Dense(1024),
                        nn.ReLU(),
                        nn.Dense(num_perms),
                    ]
                else:
                    if subtask == 'flip':
                        if flip == 0:
                            continue
                        head_channels = n_patches_per_image * flip
                    else:
                        assert subtask == 'mask'
                        if not mask:
                            continue
                        head_channels = n_patches_per_image
                    head = [
                        nn.Dense(1024),
                        nn.ReLU(),
                        nn.Dense(head_channels),
                    ]
                task_heads.append(head)

                x = shared_x
                for module in head:
                    x = module(x)
                output_xs[f'pred/{subtask}'] = x

        super().__init__(inputs=backbone.inputs, outputs=output_xs, name=name or backbone.name)
        self.backbone = backbone
        self.head = shared_head + task_heads
        self.ckpt_items = dict(
            backbone=self.backbone,
            head=self.head,
        )
