import torch
import gelib

def main(batch_size, n_channels, maxl, kernel_size, n_radius, input_size, num_classes, device):

    ############################################# Defining Stuff ####################################################
    kernel_size = [kernel_size]*3
    input_size = [input_size]*3

    # Random Input
    inputs = gelib.SO3partArr.randn(batch_size, input_size, 0, n_channels).to(device)
    labels = torch.randint(0, num_classes, size=(batch_size,)).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Define Conterpolation Matrices
    Fint = torch.randn((maxl+1)**2, n_radius, *kernel_size, 1, device = device)
    CFint = [torch.randn((maxl+1)**2, n_radius * (maxl+1), *kernel_size, 2*l+1, device = device) for l in range(maxl+1)]

    # Define Weights
    weights1 = torch.nn.ParameterList([torch.nn.Parameter(
                          torch.randn(n_channels * n_radius, n_channels, dtype = torch.cfloat, device=device))
                                       for l in range(maxl + 1)])
    weights2 = torch.nn.ParameterList([torch.nn.Parameter(
                  torch.randn(n_channels * n_radius * (maxl+1)**2, num_classes, dtype = torch.cfloat, device=device))
                                         for l in range(maxl + 1)])

    ################################################ First Layer ####################################################
    # Conterpolation
    interim1 = inputs.conterpolateB(Fint).flatten(-2)
    # the usqueeze operation introduces a new dimension at the penultimate position, which serves as a proxy for the component dimension of SO3partArr
    #the tensor is flattened afte conterpolation to collect the radius and input channels together which serve as the proxy for input channel

    # Weight Multiplication
    interim2 = gelib.SO3vecArr()
    interim2.parts = [ gelib.SO3partArr(interim1[:,:,:,:,l**2:(l+1)**2,:] @ weights1[l]) for l in range(maxl+1) ]
    # The tensor is sliced to collect the lth component

    ################################################ Second Layer #######################################################
    # Conterpolation
    interim3 = torch.cat([interim2.parts[l1].conterpolateB(CFint[l1]).flatten(-2) for l1 in range(maxl+1)], dim=-1)
    # the tensor is concated to collect all the vectors coming for different valus of l1

    # Weight Multiplication
    interim4 = gelib.SO3vecArr()
    interim4.parts = [interim3[:,:,:,:,l**2:(l+1)**2,:] @ weights2[l] for l in range(maxl+1) ]

    ################################################ Flattening #########################################################
    outputs = torch.cat(interim4.parts, dim = -2)
    outputs = torch.mean(outputs, dim = (1,2,3))
    outputs = torch.mean(outputs, dim = (1,))
    # the equivariant parts are stacked togther, averaged over the voxel locations and then the norm is calculated
    print(outputs)

    ############################################## Backpropagation ########################################################
    loss = criterion(torch.real(outputs), labels)
    loss.backward()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_channels", type=int, default=2)
    parser.add_argument("--maxl", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--n_radius", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--device", type=str, default='cuda')

    args = parser.parse_args()

    for _ in range(1):
        main(**args.__dict__)
