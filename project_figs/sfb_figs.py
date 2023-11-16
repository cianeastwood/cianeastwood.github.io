
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots import bundles

SEED = 1234

mpl.use('macOsX')
# plt.rcParams.update(bundles.icml2022())
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{times,amsmath, amsfonts}')
plt.rcParams.update(bundles.neurips2022())
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{times,amsmath, amsfonts}')
mpl.rcParams['axes.linewidth'] = 1.2  # set the value globally


def cs_plot():
    # Settings
    fontsizes = {'font.size': 18,
                 'axes.labelsize': 18,
                 'legend.fontsize': 18,
                 'xtick.labelsize': 14,
                 'ytick.labelsize': 14}
    plt.rcParams.update(fontsizes)

    # Data
    c_f = [0.98, 0.99, 0.985, 0.99]
    c_a = [1., 1., 1., 1.]
    s_f = [0., 0.581, 0.623, 0.671]
    s_a = [0., 0., 0., 0.]

    x = [5, 10, 15, 20]
    Y = [c_f, s_f, c_a, s_a]
    names = ["fixed $\lambda$: content", "fixed $\lambda$: style",
             "adaptive $\lambda$: content", "adaptive $\lambda$: content"]
    labels = ["tab:purple", "tab:purple", "tab:orange", "tab:orange"]
    linestyles = ["solid", "solid", "dashed", "dashed"]

    # Plot
    fig, ax = plt.subplots(figsize=(3.65, 3.5))

    for y, label, color, linestyle in zip(Y, names, labels, linestyles):
        sns.lineplot(x=x, y=y, ax=ax, linestyle=linestyle, 
                     color=color, label=label, linewidth=4)
    
    ax.legend(loc="best")
    # ax.legend(loc="best", ncol=3, handlelength=0.8, columnspacing=0.6,
    #           handletextpad=0.2)
    ax.set_xlabel(r"$\text{dim}(\bm{z})$")
    ax.set_ylabel("$r^2$")

    plt.savefig("ssl_cs.pdf")


def lc_curve_paper():
    # Settings
    fontsizes = {'font.size': 18,
                 'axes.labelsize': 18,
                 'legend.fontsize': 18,
                 'xtick.labelsize': 14,
                 'ytick.labelsize': 14}
    plt.rcParams.update(fontsizes)
    color = "tab:orange"
    # loss_baseline_eps = 0.07
    baseline = 0.75
    max_perf = 1.0 # 0.9
    # Create synthetic data

    correlations = np.linspace(100,-100,11)
    correlations = [100, 90, 80, 70, 60, 50, -60, -70, -80, -90, -100]
    correlations = [c / 100. for c in correlations]


    accs = np.arange(-5,6) ** 2
    accs = baseline + (max_perf - baseline) * accs / accs.max()

    capacities = np.arange(len(accs))
    # strainght line for ERM loss
    erm_loss = np.linspace(max_perf, 1.0 - max_perf, len(capacities))
    erm_loss = 2 * accs[1] - accs[2]  + capacities * (accs[2] - accs[1])

    # Plot
    # fs = figsizes.neurips2021()["figure.figsize"]
    # fs = (fs[0] * 0.85, fs[1] * 1.56)
    fs = (4.5, 2.5)
    fig, ax = plt.subplots(figsize=fs)

    # plot erm
    sns.lineplot(x=capacities, y=erm_loss, ax=ax, linestyle='-',
                 color="tab:gray", label='ERM', linewidth=4)

    # plot invariant model
    sns.lineplot(x=capacities, y=[baseline] * len(accs), ax=ax,
                 color="cornflowerblue", label='Invariant', linewidth=4)
    # plot optimal model
    sns.lineplot(x=capacities, y=accs, ax=ax, linestyle='dotted',
                 color="black",label='Oracle',markersize=5, linewidth=4)
    # add mark for training environment (90%, 80%)
    sns.lineplot(x=capacities[1:3], y=accs[1:3], ax=ax, color="limegreen", marker="o",markersize=10)

    # add mark for testing -90% environment
    sns.lineplot(x=capacities[-2:-1], y=accs[-2:-1], ax=ax, color="Red", marker="o",markersize=10)

    # Colour area under optimal curve
    xs = np.linspace(capacities[0], capacities[-1], 20)
    ys = np.interp(xs, capacities, accs)
    plt.fill_between(xs, ys, [baseline] * len(ys), color=color, alpha=0.25)

    # Add legend
    ax.legend(loc="best", ncol=3, handlelength=0.8, columnspacing=0.6,
              handletextpad=0.2)

    # Create tick labels
    c_labels = [f'{c}' for c in correlations]
    capacities = [capacities[0], capacities[5], capacities[-1]]
    c_labels = [c_labels[0], c_labels[5], c_labels[-1]]
    # Final settings
    ax.set_ylim(0.48, 1.0)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_xticks(capacities)
    ax.set_xticklabels(c_labels)
    # ax.set_xlabel("Correlation between color and label")
    ax.set_xlabel("Color-Label Correlation")

    ax.set_ylabel("Accuracy")
    # Save
    plt.savefig("optimal_curves.pdf")


def lc_curve_website(shade=True, fname="optimal_curves.pdf"):
    # Settings
    fontsizes = {'font.size': 19,
                 'axes.labelsize': 19,
                 'legend.fontsize': 17,
                 'xtick.labelsize': 19,
                 'ytick.labelsize': 19}
    plt.rcParams.update(fontsizes)
    color = "tab:orange"
    # loss_baseline_eps = 0.07
    baseline = 0.75
    max_perf = 1.0 # 0.9
    # Create synthetic data

    correlations = np.linspace(100,-100,11)
    correlations = [100, 90, 80, 70, 60, 50, -60, -70, -80, -90, -100]
    correlations = [c / 100. for c in correlations]


    accs = np.arange(-5,6) ** 2
    accs = baseline + (max_perf - baseline) * accs / accs.max()

    capacities = np.arange(len(accs))
    # strainght line for ERM loss
    erm_loss = np.linspace(max_perf, 1.0 - max_perf, len(capacities))
    erm_loss = 2 * accs[1] - accs[2]  + capacities * (accs[2] - accs[1])

    # Plot
    # fs = figsizes.neurips2021()["figure.figsize"]
    # fs = (fs[0] * 0.85, fs[1] * 1.56)
    fig, ax = plt.subplots(figsize=(3.65, 3.5))

    # plot erm
    sns.lineplot(x=capacities[:len(capacities) // 2 + 1], y=erm_loss[:len(capacities) // 2 + 1], ax=ax, linestyle='-',
                 color="tab:gray", label='ERM', linewidth=4)

    # plot invariant model
    sns.lineplot(x=capacities, y=[baseline] * len(accs), ax=ax,
                 color="cornflowerblue", label='Invariant', linewidth=4)
    # plot optimal model
    sns.lineplot(x=capacities, y=accs, ax=ax, linestyle='dotted',
                 color="black",label='Oracle',markersize=5, linewidth=4)
    # add mark for training environment (90%, 80%)
    sns.lineplot(x=capacities[1:3], y=accs[1:3], ax=ax, color="gray", marker="o",markersize=12)

    # # add mark for testing -90% environment
    # sns.lineplot(x=capacities[-2:-1], y=accs[-2:-1], ax=ax, color="Red", marker="o",markersize=10)

    if shade:
        # Colour area under optimal curve
        xs = np.linspace(capacities[0], capacities[-1], 20)
        ys = np.interp(xs, capacities, accs)
        plt.fill_between(xs, ys, [baseline] * len(ys), color=color, alpha=0.4)

    # Add legend
    ax.legend(loc="lower center", ncol=3, handlelength=0.8, columnspacing=0.6,
              handletextpad=0.2)
    ax.set_ylim(0.6, 1.02)
    ax.set_xlabel("Color-Label Correlation", labelpad=15)
    ax.set_ylabel("Accuracy")

    # Turn off axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save and pad with whitespace and make figure outer box thicker
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.15)

if __name__ == "__main__":
    np.random.seed(SEED)
    cs_plot()