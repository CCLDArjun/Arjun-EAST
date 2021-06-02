import torch

def get_indicies(map_, kernel_size):
    w = map_.shape[-1]
    map_ = torch.flatten(map_)
    len_ = map_.shape[0]

    if len_ % kernel_size != 0:
        raise Exception("Cannot get unpooling indicies on this")

    indices = []
    curr_idx = []
    i = 0
    ci = 0
    first_end = kernel_size * w
    while i + w + kernel_size <= len_:
        to_comp = torch.cat((map_[i:i + kernel_size], map_[i + w:i + w + kernel_size]), 0)
        argmax = torch.argmax(to_comp)

        if argmax < kernel_size:
            curr_idx.append(torch.argmax(to_comp) + i)
        else:
            curr_idx.append(argmax + i + w - kernel_size)

        if (i + w + kernel_size) % w == 0:
            i = i + w + kernel_size
            indices.append(curr_idx)
            curr_idx = []
        else:
            i += kernel_size
        ci += 1
    if curr_idx:
        indices.append(curr_idx)
    return torch.tensor(indices)