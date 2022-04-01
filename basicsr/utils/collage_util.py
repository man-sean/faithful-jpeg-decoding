import torch
import wandb
import numpy as np

from basicsr.utils.diffjpeg import DiffJPEG
from basicsr.utils.batch_util import expand_batch, restore_expanded_batch, get_batch


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


@run_once
def print_img_paths(dataloader, size):
    print("[Collage] Printing once")
    for idx, batch in enumerate(dataloader):
        if idx >= size:
            break
        print(f"[Collage] Adding {batch['gt_path']} to collage")
    print("[Collage] End printing")


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if image.shape[-3] == 1:
        return image

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)


def get_collage(batch, model, expansion, keep=1, create_lq_fn=lambda x: x):
    def _to_numpy(x):
        return np.einsum('...chw->...hwc', x.detach().clamp_(0, 1).cpu().numpy())

    # jpeger = DiffJPEG(differentiable=True)

    # expand batch
    inputs = ["gt", "lq", "qf"]
    xbatch = {k: expand_batch(v, expansion) for k, v in batch.items() if k in inputs}

    # generate fakes
    model.feed_data(xbatch)
    model.test()
    visuals = model.get_current_visuals()

    # extract images
    real = restore_expanded_batch(visuals['gt'], expansion)[:keep]
    compressed = restore_expanded_batch(visuals['lq'], expansion)[:keep]
    fake = restore_expanded_batch(visuals['result'], expansion)[:keep]
    # recompressed = restore_expanded_batch(jpeger(visuals['result'], quality=xbatch['qf']), expansion)[:keep]
    recompressed = restore_expanded_batch(create_lq_fn(visuals['result']), expansion)[:keep]
    diff = (compressed - recompressed).abs()

    # remove redundancies
    real = real[:1].squeeze(0)
    compressed = compressed[:1].squeeze(0)
    fake = fake.squeeze(0)
    recompressed = recompressed.squeeze(0)
    diff = diff.squeeze(0)

    # generate std
    std_fake = visuals['result'].clamp(0, 1)
    std_fake = rgb_to_ycbcr(std_fake)[..., 0:1, :, :]
    std = restore_expanded_batch(std_fake, expansion).std(0)
    caption = [f'STD stats: {std[idx].mean():.4} ±{std[idx].std():.4} [{std[idx].min():.3}, {std[idx].max():.3}]'
               for idx in range(std.shape[0])]
    std = std.pow(1 / 4).expand_as(real).clone()
    std = 1 - _to_numpy(std)

    # convert to numpy
    real = _to_numpy(real)
    compressed = _to_numpy(compressed)
    fake = _to_numpy(fake)
    recompressed = _to_numpy(recompressed)
    diff = _to_numpy(diff)

    return real, compressed, fake, recompressed, diff, std, caption


def log_collage(dataloader, model, global_step, size=10, expansion=32):
    real, compressed, fake, recompressed, diff, std, caption = [], [], [], [], [], [], []

    print_img_paths(dataloader=dataloader, size=size)  # will run only once

    for idx, batch in enumerate(dataloader):
        if idx >= size:
            break
        out = get_collage(batch, model, expansion, create_lq_fn=dataloader.dataset.create_lq)
        real[len(real):], compressed[len(compressed):], fake[len(fake):], \
        recompressed[len(recompressed):], diff[len(diff):], std[len(std):], \
        caption[len(caption):] = tuple(zip(out))

    wandb.log(
        {"collage": [wandb.Image(np.concatenate([r[0], c[0], f[0], rc[0], d[0], s[0]], axis=1), caption=t)
                     for r, c, f, rc, d, s, t in zip(real, compressed, fake, recompressed, diff, std, caption)],
         "global_step": global_step}
    )
