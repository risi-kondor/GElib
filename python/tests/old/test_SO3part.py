import torch
import gelib as G
import pytest

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_init_so3part(device):
    if not torch.cuda.is_available():
        return
    a = torch.randn(3, 5, 3, 2, device=device)
    a_prt = G.SO3part(a)
