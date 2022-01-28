import torch


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