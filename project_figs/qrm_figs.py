import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from tueplots import bundles


# Global settings
N_SAMPLES = 10
X_RANGE = [0, 1.2]
# Y_RANGE = [-0.05, 1.02]
Y_RANGE = [-0.05, 1.05]
EVEN_SPACING = False
SEED = 1234

plt.rcParams.update(bundles.neurips2022())
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{times,amsmath, amsfonts}')


def coloured_cdf():
    # Generate samples
    if EVEN_SPACING:
        xs = np.linspace(0.2, 0.6, N_SAMPLES-1)
    else:
        xs = np.random.uniform(0.1, 0.6, size=N_SAMPLES - 1)
    xs = np.array(list(sorted(xs)) + [1.0])

    # Plot
    fig, ax = plt.subplots()

    # Empirical cdf
    sns.histplot(xs, element="step", fill=False, cumulative=True, binwidth=0.01, binrange=X_RANGE,
                 color="black", label="Empirical", stat="density", ax=ax)

    # Smoothed cdf (KDE)
    kde = utils.KernelDensityEstimator(torch.from_numpy(xs), bandwidth_est_method="Gauss-optimal", alpha=0.5)
    xs_ = torch.linspace(*X_RANGE, 1000)
    cum_p_xs = [kde.cdf(x) for x in xs_]
    ax.plot(xs_, cum_p_xs, label="KDE", color="gray", linestyle="dashed")

    # Colour background
    upper_ys = [0.45, 0.55, 0.98, 1., 1.]                           # alphas
    upper_xs = [np.quantile(xs, a) for a in upper_ys]
    upper_xs[-2] += 0.05
    cmap = plt.cm.coolwarm_r(np.linspace(0, 1, len(upper_ys) + 1))
    # print([matplotlib.colors.to_hex(c) for c in cmap])
    for i, (x, y) in enumerate(zip(upper_xs[::-1], upper_ys[::-1])):
        if i == 0:
            ax.axhspan(Y_RANGE[0], Y_RANGE[1], facecolor=cmap[i])
        elif i == (len(upper_ys) - 1):
            ax.axhspan(Y_RANGE[0], y, xmax=x / X_RANGE[1], facecolor=cmap[-1])
        else:
            ax.axhspan(Y_RANGE[0], y, xmax=x / X_RANGE[1], facecolor=cmap[i])

    # Samples
    ax.scatter(xs, np.zeros(len(xs)) - 0.025, color="black")

    # Final settings
    ax.set_ylim(*Y_RANGE)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Risk $r$")
    ax.set_ylabel(r"$F(r)$")
    plt.legend()
    plt.savefig("cdf.svg")


def risk_dist_pdf_cdf(alpha=0.9, plot_kde_bumps=False):
    # Settings
    pdf_color = "tab:blue"
    kde_color = "gray"
    cdf_color = "black"
    sample_color = cdf_color
    alpha_color = "tab:red"
    cdf_on_right = False

    # Settings
    fontsizes = {'font.size': 10,
                 'axes.labelsize': 10,
                 'legend.fontsize': 10,
                 'xtick.labelsize': 8,
                 'ytick.labelsize': 8}

    # Generate samples
    if EVEN_SPACING:
        xs = np.linspace(0.2, 0.6, N_SAMPLES - 1)
    else:
        xs = np.random.uniform(0.1, 0.6, size=N_SAMPLES - 1)
    xs = np.array(list(sorted(xs)) + [1.0])
    xs_ = torch.linspace(*X_RANGE, 1000)

    # Fit KDE
    kde = utils.KernelDensityEstimator(torch.from_numpy(xs), bandwidth_est_method="silverman", alpha=0)
    p_xs = kde(xs_)
    cum_p_xs = [kde.cdf(x) for x in xs_]

    # Plot samples and set up two axes (for pdf and cdf)
    fig, ax1 = plt.subplots(1, figsize=(3.65, 3))
    ax1.scatter(xs, np.zeros(len(xs)) - 0.025, color=sample_color, s=8)
    ax1.set_xlabel("Risk")
    ax1.set_xticks([0., 0.5, 1.0])
    ax2 = ax1.twinx()       # instantiate a second axes that shares the same x-axis

    # Plot kernel bumps
    if plot_kde_bumps:
        for mu in xs:
            xs_normal = torch.linspace(mu - 0.25, mu + 0.25, 100)
            p_xs_normal = torch.exp(torch.distributions.Normal(mu, kde.kernel.bandwidth).log_prob(xs_normal)).numpy()
            p_xs_normal_kde = p_xs_normal / len(xs)
            ax1.plot(xs_normal, p_xs_normal_kde, color=kde_color, alpha=0.15)  # alpha is opaqueness...

    def plot_pdf(pdf_ax):
        pdf_ax.plot(xs_, p_xs, label="PDF", color=pdf_color, linewidth=2)
        pdf_ax.set_ylabel(r"Density", color=pdf_color)
        pdf_ax.tick_params(axis='y', labelcolor=pdf_color)
        pdf_ax.set_ylim(Y_RANGE[0])
        pdf_ax.set_yticks([0., 0.5, 1.0, 1.5])

    def plot_cdf(cdf_ax):
        cdf_ax.plot(xs_, cum_p_xs, label="CDF", color=cdf_color, linewidth=2)
        cdf_ax.set_ylabel('Probability', color=cdf_color)  # we already handled the x-label with first ax
        cdf_ax.tick_params(axis='y', labelcolor=cdf_color)
        cdf_ax.set_ylim(*Y_RANGE)
        cdf_ax.set_yticks([0., 0.5, 1.0])

    def plot_alpha_lines(alpha_ax, cdf_on_right=True):
        t_alpha = kde.icdf(alpha)
        if cdf_on_right:
            alpha_ax.axhline(y=alpha, xmin=t_alpha / X_RANGE[1], color=alpha_color, linestyle="dashed")     # horizontal
            alpha_ax.axvline(x=t_alpha, ymax=alpha / Y_RANGE[1], ymin=-0.5, color=alpha_color, linestyle="dashed")  # vertical

            # Annotate alpha lines
            alpha_ax.annotate(r'$\boldsymbol{\alpha}$', xy=(1.01, alpha / Y_RANGE[1] - 0.01),
                              xycoords='axes fraction', color=alpha_color)
            alpha_ax.annotate(r'$F^{-1}(\boldsymbol{\alpha})$', xy=(t_alpha / (X_RANGE[1] + 0.11), -0.05),  # -0.115
                              xycoords='axes fraction', color=alpha_color)
        else:
            alpha_ax.axhline(y=alpha, xmax=t_alpha / X_RANGE[1] - 0.02, color=alpha_color, linestyle="dashed")  # horizontal
            alpha_ax.axvline(x=t_alpha, ymax=alpha / Y_RANGE[1], ymin=-0.5, color=alpha_color,
                             linestyle="dashed")  # vertical

            # Annotate alpha lines
            alpha_ax.annotate(r'$\boldsymbol{\alpha}$', xy=(-0.06, alpha / Y_RANGE[1] - 0.005),
                              xycoords='axes fraction', color=alpha_color)
            # alpha_ax.annotate(r'$F^{-1}(\boldsymbol{\alpha})$', xy=(t_alpha / (X_RANGE[1] + 0.11), -0.115),
            #                   xycoords='axes fraction', color=alpha_color)
            alpha_ax.annotate(r'$\boldsymbol{\alpha}$-quantile', xy=(t_alpha / (X_RANGE[1] + 0.3), -0.05),  # -0.115
                              xycoords='axes fraction', color=alpha_color)

    # Plot pdf and cdf
    if cdf_on_right:
        pdf_ax, cdf_ax = ax1, ax2
    else:
        pdf_ax, cdf_ax = ax2, ax1

    plot_pdf(pdf_ax)
    plot_cdf(cdf_ax)
    plot_alpha_lines(cdf_ax, cdf_on_right)

    handles, _ = ax1.get_legend_handles_labels()
    handles2, _ = ax2.get_legend_handles_labels()
    handles.extend(handles2)

    # Final settings and save
    plt.legend(handles=handles, loc="center right")
    plt.tight_layout()
    plt.savefig("risk_dist_pdf_cdf_left_new.svg")


def risk_dist_cdf_steps(alpha=0.98):
    # fontsizes = {'font.size': 13,
    #              'axes.labelsize': 14,
    #              'legend.fontsize': 14,
    #              'xtick.labelsize': 8,
    #              'ytick.labelsize': 8}
    fontsizes = {'font.size': 12,
                 'axes.labelsize': 12,
                 'legend.fontsize': 12,
                 'xtick.labelsize': 10,
                 'ytick.labelsize': 10}
    plt.rcParams.update(fontsizes)


    # Settings
    step_color = "gray"
    cdf_color = "black"
    sample_color = cdf_color
    alpha_color = "tab:red"

    # Generate samples
    if EVEN_SPACING:
        xs = np.linspace(0.2, 0.6, N_SAMPLES - 1)
    else:
        xs = np.random.uniform(0.1, 0.6, size=N_SAMPLES - 1)
    xs = np.array(list(sorted(xs)) + [1.0])
    xs_ = torch.linspace(*X_RANGE, 1000)

    # Fit KDE
    kde = utils.KernelDensityEstimator(torch.from_numpy(xs), bandwidth_est_method="silverman", alpha=0)
    cum_p_xs = [kde.cdf(x) for x in xs_]

    # Plot samples and set up two axes (for pdf and cdf)
    fig, ax1 = plt.subplots(1, figsize=(3.65, 3))
    ax1.scatter(xs, np.zeros(len(xs)) - 0.025, color=sample_color, s=10)
    ax1.set_xlabel("Risk")
    ax1.set_xticks([0., 0.5, 1.0])
    ax1.yaxis.set_tick_params(pad=7.5)
    ax1.xaxis.set_tick_params(pad=4)

    ax1.plot(xs_, cum_p_xs, label="KDE", color=cdf_color, linewidth=3)
    ax1.set_ylabel('Probability', color=cdf_color)  # we already handled the x-label with first ax
    ax1.tick_params(axis='y', labelcolor=cdf_color)
    ax1.set_ylim(*Y_RANGE)
    ax1.set_yticks([0., 0.5, 1.0])
    t_alpha = kde.icdf(alpha)
    ax1.axhline(y=alpha, xmax=t_alpha / X_RANGE[1] - 0.05, color=alpha_color, linestyle="dashed", linewidth=2.5)  # horizontal
    ax1.axvline(x=t_alpha, ymax=alpha / Y_RANGE[1], ymin=-0.5, color=alpha_color,
                linestyle="dashed", linewidth=2.5)  # vertical

    # Annotate alpha lines
    ax1.annotate(r'$\boldsymbol{\alpha}$', xy=(-0.05, alpha / Y_RANGE[1] - 0.0175),
                 xycoords='axes fraction', color=alpha_color)
    # alpha_ax.annotate(r'$F^{-1}(\boldsymbol{\alpha})$', xy=(t_alpha / (X_RANGE[1] + 0.11), -0.115),
    #                   xycoords='axes fraction', color=alpha_color)
    ax1.annotate(r'$F^{-1}(\boldsymbol{\alpha})$', xy=(t_alpha / (X_RANGE[1] + 0.1), -0.065),  # -0.115
                 xycoords='axes fraction', color=alpha_color)

    # Plot Empirical cdf
    sns.histplot(xs, element="step", fill=False, cumulative=True, binwidth=0.01, binrange=X_RANGE,
                 color=step_color, label="Empirical", stat="density", ax=ax1, linewidth=3, alpha=0.75)

    # Final settings and save
    # plt.legend(loc="best", handlelength=1.5)
    plt.legend(loc='lower center', handlelength=0.75, handletextpad=0.65, bbox_to_anchor=(0.5, 0.025))
    # plt.tight_layout()
    plt.savefig("risk_dist_cdf_step.pdf")


if __name__ == "__main__":
    np.random.seed(SEED)

    # coloured_cdf()
    # risk_dist_pdf_cdf()
    risk_dist_cdf_steps()
