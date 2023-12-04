import torch

def fft(img: torch.Tensor
        ) -> torch.Tensor:
    img_new = torch.fft.ifftshift(img, dim = (-2, -1))
    img_new = torch.fft.fftn(img_new, dim = (-1, -2), norm = 'ortho')
    img_new = torch.fft.fftshift(img_new, dim = (-2, -1))

    return img_new

def ifft(img: torch.Tensor
         ) -> torch.Tensor:
    img_new = torch.fft.ifftshift(img, dim = (-2, -1))
    img_new = torch.fft.ifftn(img_new, dim = (-2, -1))
    img_new = torch.fft.fftshift(img_new, dim = (-2, -1))
    return img_new


def complex2channel(img: torch.Tensor
                    ) -> torch.Tensor:
    
    N, C, H, W= img.shape
    img = torch.view_as_real(img)
    img = img.permute(0, 1, 4, 2, 3).contiguous()
    return img.view(N, C*2, H, W)