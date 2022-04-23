import torch
import gelib as G
import numpy as np
import matplotlib.pyplot as plt


def test_equivariance(x, maxl):
    # x = G.SO3vec.randn(batch_size, [tau for i in range(maxl + 1)])
    x_rot = G.SO3vec.zeros_like(x)
    R = G.SO3element.uniform()
    for i in range(len(x.parts)):
        x_rot.parts[i] =x.parts[i].detach().apply(R)


    x_out = G.DiagCGproduct(x, x, maxl=maxl)
    x_rot_out = G.DiagCGproduct(x_rot, x_rot, maxl=maxl)

    x_out_rot = G.SO3vec.zeros_like(x_out)

    diffs = []
    for i in range(maxl+1):
        x_out_rot.parts[i] = x_out.parts[i].detach().apply(R)
        diff = x_out_rot.parts[i] - x_rot_out.parts[i]
        diffs.append(diff)
    return diffs


def main():
    maxl = 29
    input_l= 15
    # fig, axes = plt.subplots(1, 1+maxl, figsize=((1+maxl) * 2, 2))
    fig, axes = plt.subplots(5, 6, figsize=(12, 10))
    axf = np.array(axes).ravel()
    for i in range(1):
        # x = G.SO3vec.randn(1, [1 for i in range(input_l+ 1)])
        x = G.SO3vec.randn(1, [1 for i in range(input_l+ 1)])

        diff = test_equivariance(x, maxl)
        for l, dif_l in enumerate(diff):
            dif_norm = torch.linalg.norm(dif_l,dim=-1)
            dif_norm = torch.linalg.norm(dif_norm, ord=1, dim=1)
            dif_norm = torch.linalg.norm(dif_norm, dim=0).detach().numpy()
            print(l)
            print(dif_norm)
            print("\n")
            #axf[l].plot(np.arange(len(dif_norm)), dif_norm)

    for l in range(maxl+1):
        axf[l].set_xlabel("channel")
        axf[l].set_ylabel("resid")
        axf[l].set_title('l=%d' % l)
        axf[l].set_yscale('log')
    plt.tight_layout()
    plt.savefig("image_resid_in_%d_out_%d.png" % (input_l, maxl))
    # plt.show()


if __name__ == "__main__":
    main()
