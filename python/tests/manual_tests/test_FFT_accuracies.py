import torch
import gelib
torch.manual_seed(123)

batch_size = 200
maxl = 5
device = 0

signals = gelib.SO3vec.Frandn(batch_size, maxl, device)

for bandwidth in range(1, 101):
    inverse = gelib.SO3iFFT(signals, bandwidth)
    signals_ = gelib.SO3FFT(inverse, maxl)

    sum_errors = 0
    total_elems = 0
    for l in range(len(signals.parts)):
        error = torch.sum(torch.abs(torch.tensor(signals.parts[l]) - torch.tensor(signals_.parts[l])))
        print(torch.tensor(error))
        total_elems += signals.parts[l].numel()
    average_error = sum_errors / total_elems
    print(bandwidth, sum_errors, average_error)

