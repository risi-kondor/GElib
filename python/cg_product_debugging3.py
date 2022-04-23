import torch
import gelib as G
import numpy as np
import matplotlib.pyplot as plt


def test_equivariance(x, y, maxl):
    # x = G.SO3vec.randn(batch_size, [tau for i in range(maxl + 1)])
    x_rot = G.SO3vec.zeros_like(x)
    y_rot = G.SO3vec.zeros_like(y)
    R = G.SO3element.uniform()
    for i in range(len(x.parts)):
        x_rot.parts[i] =x.parts[i].detach().apply(R)

    for i in range(len(y.parts)):
        y_rot.parts[i] =y.parts[i].detach().apply(R)


    x_out = G.DiagCGproduct(x, y, maxl=maxl)
    x_rot_out = G.DiagCGproduct(x_rot, y_rot, maxl=maxl)

    x_out_rot = G.SO3vec.zeros_like(x_out)
    # print([len(p) for p in x_out_rot.parts])
    # print(len([len(p) for p in x_rot_out.parts]))
    # raise Exception

    diffs = []
    for i in range(maxl+1):
        x_out_rot.parts[i] = x_out.parts[i].detach().apply(R)
        diff = x_out_rot.parts[i] - x_rot_out.parts[i]
        diffs.append(diff)
    return diffs, x_rot_out


def make_plot_of_errs():
    maxl = 29
    input_l= 15
    # maxl = 5
    # input_l = 3
    # maxl = 20
    # input_l= 10
    # fig, axes = plt.subplots(1, 1+maxl, figsize=((1+maxl) * 2, 2))
    # fig, axes = plt.subplots(1, 1+maxl, figsize=((1+maxl) * 2, 2))
    fig, axes = plt.subplots(5, 6, figsize=(12, 10))
    axf = np.array(axes).ravel()
    taus = [0] * input_l
    taus[-1] = 1
    print(taus)
    for i in range(10):
        x = G.SO3vec.randn(1, [1 for i in range(input_l+ 1)])
        y = G.SO3vec.randn(1, [1 for i in range(input_l+ 1)])
        # x = G.SO3vec.randn(1, taus)

        diff, x_out_rot = test_equivariance(x, y, maxl)
        for l, (dif_l, x_or_l) in enumerate(zip(diff, x_out_rot.parts)):
            # print(l, torch.view_as_complex(x_or_l))
            print(dif_l.shape)
            dif_norm = torch.linalg.norm(dif_l, dim=-1)
            dif_norm = torch.linalg.norm(dif_norm, dim=1)
            dif_norm = torch.linalg.norm(dif_norm, dim=0).detach().numpy()

            x_or_nrm = torch.linalg.norm(x_or_l, dim=-1)
            x_or_nrm = torch.linalg.norm(x_or_nrm, dim=1)
            x_or_nrm = torch.linalg.norm(x_or_nrm, dim=0)
            dif_norm /= x_or_nrm
            axf[l].plot(np.arange(len(dif_norm)), dif_norm)
            #print(l)
            #print(dif_norm)
            print(len(dif_norm))

    for l in range(maxl+1):
        axf[l].set_xlabel("channel")
        axf[l].set_ylabel("resid")
        axf[l].set_title('l=%d' % l)
        axf[l].set_yscale('log')
        axf[l].set_ylim(1e-7, 1e1)
    plt.tight_layout()
    plt.savefig("image_resid_in_%d_out_%d_sparse.png" % (input_l, maxl))
    plt.show()

def print_output():
    maxl = 6
    input_l = 3
    for i in range(1):
        in_taus = [0] * (input_l + 1)
        in_taus[input_l] = 1
        x = G.SO3vec.randn(1, in_taus)
        y = G.SO3vec.randn(1, in_taus)
        print(x)
        # x = G.SO3vec.randn(1, [1 for i in range(input_l+ 1)])
        x_out = G.DiagCGproduct(x, y, maxl=maxl)
        print(x_out)

        # for l, part in enumerate(x_out.parts):
        #     print("--------------------")
        #     print(l)
        #     print(torch.view_as_complex(part).squeeze(0))

def main():
    # print_output()
    make_plot_of_errs()


if __name__ == "__main__":
    main()
