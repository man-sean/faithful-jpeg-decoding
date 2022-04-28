import itertools

import numpy as np
import torch
import torch.nn as nn


class BlockSplitting(nn.Module):
    """ Splitting image into patches of size k x k
    """

    def __init__(self, k=8):
        super(BlockSplitting, self).__init__()
        self.k = k

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor:  batch x h*w/64 x h x w
        """
        height, _ = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)


class DCT(nn.Module):
    """ Discrete Cosine Transformation
    """

    def __init__(self, n=8):
        super(DCT, self).__init__()
        tensor = np.zeros((n, n, n, n), dtype=np.float32)
        for x, y, u, v in itertools.product(range(n), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / (2 * n)) * np.cos((2 * y + 1) * v * np.pi / (2 * n))
        alpha = np.array([2. / np.sqrt(4 * n)] + [2. / np.sqrt(2 * n)] * (n-1))
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class iDCT(nn.Module):
    """Inverse discrete Cosine Transformation
    """

    def __init__(self, n=8):
        super(iDCT, self).__init__()
        alpha = np.array([2. / np.sqrt(4 * n)] + [2. / np.sqrt(2 * n)] * (n-1))
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((n, n, n, n), dtype=np.float32)
        for x, y, u, v in itertools.product(range(n), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / (2 * n)) * np.cos((2 * v + 1) * y * np.pi / (2 * n))
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        image = image * self.alpha
        result = torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class BlockMerging(nn.Module):
    """Merge patches of size k x k into image
    """

    def __init__(self, k=8):
        super(BlockMerging, self).__init__()
        self.k = k

    def forward(self, patches, height, width):
        """
        Args:
            patches(tensor) batch x height*width/64, height x width
            height(int)
            width(int)

        Returns:
            Tensor: batch x height x width
        """
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height // self.k, width // self.k, self.k, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)