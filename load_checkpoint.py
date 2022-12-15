# Evaluate saved checkpoint
# Checkpoint loading

import timm
import torch

import numpy as np
from scipy import interpolate

import modeling_finetune  # Models are auto-registered with the import
from config import Config


class args:
    # Model creation
    model = 'beit_base_patch16_224'
    pretrained = False
    nb_classes = 2
    drop = 0
    drop_path = .1
    attn_drop_rate = 0
    drop_block_rate = None
    use_mean_pooling = True
    init_scale = .001
    rel_pos_bias = True
    abs_pos_emb = False
    layer_scale_init_value = .1

    # Miscellaneous
    input_size = Config.image_size
    model_prefix = ''


class LoadCheckpoint:
    def __init__(self, checkpoint_path=None):
            self.checkpoint_path = checkpoint_path
            self.classes = 2 if Config.ce_loss else 1

    def load_checkpoint(self):
        # args = ModelArgs

        model = timm.models.create_model(
            args.model,
            pretrained=False,
            num_classes=self.classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
        )

        patch_size = model.patch_embed.patch_size
        print("Patch size = %s" % str(patch_size))
        args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
        args.patch_size = patch_size

        print("Load ckpt from %s" % self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        checkpoint_model = checkpoint['model']

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if model.use_rel_pos_bias and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
            print("Expand the shared relative position embedding to each transformer block. ")
            num_layers = model.get_num_layers()
            rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
            for i in range(num_layers):
                checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

            checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

            if "relative_position_bias_table" in key:
                rel_pos_bias = checkpoint_model[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = model.state_dict()[key].size()
                dst_patch_shape = model.patch_embed.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                if src_size != dst_size:
                    print("Position interpolate for %s from %dx%d to %dx%d" % (
                        key, src_size, src_size, dst_size, dst_size))
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                    checkpoint_model[key] = new_rel_pos_bias

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        # This loads the state dict into memory with the same object
        load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        return model

    def hammer_time(self):
        model = self.load_checkpoint()
        return model


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))
