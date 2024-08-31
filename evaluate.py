import numpy as np
from matplotlib import pyplot as plt
import os
import torch

from train import Problem

FIGURES_DIR = "figs"

def get_result_name(pb: Problem, use_encoder: bool = None):
    if use_encoder is None:
        model_name = "mixed"
    elif use_encoder:
        model_name = "autoencoder"
    else:
        model_name = "decoder"
    return "%s_%s_noise%s" % (model_name, pb.name, pb.noise_std)

def get_figure_path(pb: Problem, use_encoder, extension=".png"):
    name = get_result_name(pb, use_encoder)
    return os.path.join(FIGURES_DIR, name + extension)

def eval_errors(xs, xs_predict, transient_len):
    print("Evaluating errors...")
    print("xs shape:", xs.shape)
    print("xs_predict shape:", xs_predict.shape)
    #errs = np.sqrt(((xs - xs_predict.numpy()) ** 2).mean(-1).mean(0))
    errs = np.sqrt(((xs - xs_predict) ** 2).mean(-1).mean(0))
    tot = errs.mean()
    transient = errs[:transient_len].mean()
    asymptotic = errs[transient_len:].mean()
    print("errs: total %f transient %f asymptotic %f" %
          (tot, transient, asymptotic))

def render_duffing(pb, ts, ys, xs, xs_predict, idx, transient_len, use_encoder):
    eval_errors(xs, xs_predict, transient_len)

    # -- plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    
    ax[0].plot(ts, xs[idx, :, 0], 'b-', label='$x_1$')
    ax[0].plot(ts, xs_predict[idx, :, 0], 'r--', label='$\hat{x}_1$')
    ax[0].set_ylabel('State $x_1$')
    ax[0].legend(loc=1)

    
    ax[1].plot(ts, xs[idx, :, 1], 'b-', label='$x_2$')
    ax[1].plot(ts, xs_predict[idx, :, 1], 'r--', label='$\hat{x}_2$')
    ax[1].set_ylabel('State $x_2$')
    ax[1].set_xlabel('time')
    ax[1].legend(loc=1)

    plt.tight_layout()
    
    # -- save figure
    fig_path = get_figure_path(pb, use_encoder, extension=".png")
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print("Figure saved:", fig_path)
