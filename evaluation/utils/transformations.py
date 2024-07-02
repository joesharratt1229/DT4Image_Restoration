import torch
from typing import Optional

def fft(img: torch.Tensor
        ) -> torch.Tensor:
    img_new = torch.fft.ifftshift(img, dim = (-2, -1))
    img_new = torch.fft.fftn(img_new, dim = (-2, -1), norm = 'ortho')
    img_new = torch.fft.fftshift(img_new, dim = (-2, -1))

    return img_new

def ifft(img: torch.Tensor
         ) -> torch.Tensor:
    img_new = torch.fft.ifftshift(img, dim = (-2, -1))
    img_new = torch.fft.ifftn(img_new, dim = (-2, -1), norm = 'ortho')
    img_new = torch.fft.fftshift(img_new, dim = (-2, -1))
    return img_new


def spi_inverse(ztilde, K1, K, mu):
    """
    Proximal operator "Prox\_{\frac{1}{\mu} D}" for single photon imaging
    assert alpha == K and q == 1
    """
    z = torch.zeros_like(ztilde)

    K0 = K**2 - K1
    indices_0 = (K1 == 0)

    z[indices_0] = ztilde[indices_0] - (K0 / mu)[indices_0]

    func = lambda y: K1 / (torch.exp(y) - 1) - mu * y - K0 + mu * ztilde

    indices_1 = torch.logical_not(indices_0)

    # differentiable binary search
    bmin = 1e-5 * torch.ones_like(ztilde)
    bmax = 1.1 * torch.ones_like(ztilde)

    bave = (bmin + bmax) / 2.0

    for i in range(10):
        tmp = func(bave)
        indices_pos = torch.logical_and(tmp > 0, indices_1)
        indices_neg = torch.logical_and(tmp < 0, indices_1)
        indices_zero = torch.logical_and(tmp == 0, indices_1)
        indices_0 = torch.logical_or(indices_0, indices_zero)
        indices_1 = torch.logical_not(indices_0)

        bmin[indices_pos] = bave[indices_pos]
        bmax[indices_neg] = bave[indices_neg]
        bave[indices_1] = (bmin[indices_1] + bmax[indices_1]) / 2.0

    z[K1 != 0] = bave[K1 != 0]
    return torch.clamp(z, 0.0, 1.0)