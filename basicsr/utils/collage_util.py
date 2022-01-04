import torch
import wandb
import numpy as np

from basicsr.utils.diffjpeg import DiffJPEG


def get_batch(dataloader, num_samples):
    iterator = iter(dataloader)
    final_batch = next(iterator)

    def get_batch_size(batch):
        if type(batch) is dict:
            return list(batch.values())[0].shape[0]
        return batch.shape[0]

    def append_to_batch(batch, new_elems):
        if type(batch) is dict:
            for key, value in batch.items():
                try:
                    if isinstance(value, list):
                        batch[key].append(new_elems[key])
                    else:
                        batch[key] = torch.cat((value, new_elems[key]), dim=0)
                except TypeError as e:
                    raise Exception(f'key "{key}" generated an error') from e
            return batch
        return torch.cat((batch, new_elems), dim=0)

    def shrink_batch(batch, new_batch_size):
        if type(batch) is dict:
            for key, value in batch.items():
                batch[key] = value[:new_batch_size]
            return batch
        return batch[:new_batch_size]

    while get_batch_size(final_batch) < num_samples:
        final_batch = append_to_batch(final_batch, next(iterator))
    final_batch = shrink_batch(final_batch, num_samples)
    return final_batch


def reshape_batch(batch):
    new_shape = [batch.shape[0] * batch.shape[1]] + list(batch.shape[2:])
    return batch.contiguous().view(*new_shape)


def expand_batch(batch, n):
    if n == 0:
        return batch
    new_shape = [n] + [-1] * len(batch.shape)
    return reshape_batch(batch.unsqueeze(0).expand(*new_shape))


def restore_expanded_batch(expanded_batch, n):
    if n == 0:
        return expanded_batch
    new_shape = [n, expanded_batch.shape[0] // n] + list(expanded_batch.shape[1:])
    return expanded_batch.contiguous().view(*new_shape)


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


def log_collage(dataloader, model, global_step, size=10):
    def _to_numpy(x):
        return np.transpose(x.detach().clamp_(0, 1).cpu().numpy(), (0, 2, 3, 1))

    batch = get_batch(dataloader, size)
    jpeger = DiffJPEG(differentiable=True)
    expansion = 32  # number of realizations for std computation

    # generate fakes
    model.feed_data(batch)
    model.test()
    visuals = model.get_current_visuals()
    real = visuals['gt']
    compressed = visuals['lq']
    fake = visuals['result']
    recompressed = jpeger(fake, quality=batch['qf'])
    diff = (compressed - recompressed).abs()

    # generate std
    inputs = ["gt", "lq", "qf"]
    std_batch = {k: expand_batch(v, expansion) for k, v in batch.items() if k in inputs}
    model.feed_data(std_batch)
    model.test()
    visuals = model.get_current_visuals()
    std_fake = visuals['result'].clamp(0, 1)
    std_fake = rgb_to_ycbcr(std_fake)[..., 0:1, :, :]
    std = restore_expanded_batch(std_fake, expansion).std(0).pow(1/4)
    caption = f'STD min: {std.min():.3}, max: {std.max():.3}'
    std = std.expand_as(fake).clone()
    std = 1 - _to_numpy(std)

    # convert to numpy
    real = _to_numpy(real)
    compressed = _to_numpy(compressed)
    fake = _to_numpy(fake)
    recompressed = _to_numpy(recompressed)
    diff = _to_numpy(diff)

    wandb.log(
        {"collage": [wandb.Image(np.concatenate([r, c, f, rc, d, s], axis=1), caption=caption)
                     for r, c, f, rc, d, s in zip(real, compressed, fake, recompressed, diff, std)],
         "global_step": global_step}
    )
