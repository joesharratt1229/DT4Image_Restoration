import torch
import numpy as np
from scipy.ndimage import gaussian_filter


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



def calculate_ssim(img1, img2, k1=0.01, k2=0.03, win_size=11, L=255):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    
    Parameters:
    - img1, img2: Input images (2D numpy arrays)
    - k1, k2: Constants for stability
    - win_size: Size of the Gaussian window
    - L: Dynamic range of pixel values (typically 2^(#bits per pixel) - 1)
    
    Returns:
    - ssim_map: SSIM score for each pixel
    - ssim_score: Mean SSIM score
    """
    
    # Constants
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    # Compute means
    mu1 = gaussian_filter(img1, sigma=1.5, truncate=win_size//2)
    mu2 = gaussian_filter(img2, sigma=1.5, truncate=win_size//2)
    
    # Compute variances and covariance
    sigma1_sq = gaussian_filter(img1**2, sigma=1.5, truncate=win_size//2) - mu1**2
    sigma2_sq = gaussian_filter(img2**2, sigma=1.5, truncate=win_size//2) - mu2**2
    sigma12 = gaussian_filter(img1 * img2, sigma=1.5, truncate=win_size//2) - mu1 * mu2
    
    # Compute SSIM
    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / den
    
    # Return SSIM map and mean SSIM
    return ssim_map, np.mean(ssim_map)