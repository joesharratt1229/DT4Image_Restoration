from typing import Optional, Tuple
import numpy as np
from numpy.lib.stride_tricks import as_strided

def generate_radial_mask(acceleration : int,
                         shape : Tuple[int, int] = (128, 128),
                         num_angles : Optional[int] = None) -> np.ndarray:
    
    mask = np.zeros(shape)

    cy, cx = shape[0] // 2, shape[1] // 2
    
    # Calculate the number of angles if not provided
    if num_angles is None:
        num_angles = (shape[0] // acceleration) * 2

    
    for i in range(num_angles):
        angle = 2*np.pi * i/num_angles

        #compute radial line coordinates
        x = np.cos(angle) * np.arange(-cy, cy)
        y = np.sin(angle) * np.arange(-cx, cx)

        x_idx = np.round(x + cx).astype(int)
        y_idx = np.round(y + cy).astype(int)

        mask_idx = (x_idx >= 0) & (x_idx < shape[0]) & (y_idx >= 0) & (y_idx < shape[1])
        x_idx, y_idx = x_idx[mask_idx], y_idx[mask_idx]
        
        # Set corresponding mask locations to 1
        mask[y_idx, x_idx] = 1

    return mask

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask(acc, sample_n=10, shape = (128, 128)):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..

    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    return mask


def variable_density_mask(shape = (128, 128), max_density=1.0, min_density=0.1):
    """
    Create a variable density mask for a 2D grid.
    
    Parameters:
    - shape: tuple, shape of the 2D grid (height, width).
    - max_density: float, maximum sampling density at the center.
    - min_density: float, minimum sampling density at the outer region.
    """
    assert len(shape) == 2, "Shape must be 2D (height, width)"
    
    # Generate coordinates for the center of the image
    center_x, center_y = shape[1] / 2, shape[0] / 2
    
    # Create a meshgrid of coordinates
    y, x = np.ogrid[:shape[0], :shape[1]]
    
    # Compute the distance from the center for each coordinate
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Normalize the distance to the range [0, 1]
    normalized_distance = distance_from_center / np.max(distance_from_center)
    
    # Generate probability values based on a linear transition between max_density and min_density
    probabilities = max_density * (1 - normalized_distance) + min_density * normalized_distance
    
    # Sample the mask using the probabilities
    mask = np.random.uniform(0, 1, shape) < probabilities
    
    return mask.astype(np.int)
