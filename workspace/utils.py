from matplotlib import cm, colors, pyplot, ticker
pyplot.rcParams.update({'font.size': 14})


def plot_curves( x, Y, ylabel, filename, xlabel='Time', labels=None ):

    pyplot.figure(figsize=(6.4, 4.8), dpi=150)
    if labels == None:
        for y in Y:
            pyplot.plot(x, y)
    else:
        for y, label in zip(Y, labels):
            pyplot.plot(x, y, label=label)
        pyplot.legend()
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.tight_layout()
    pyplot.savefig(filename)
    pyplot.close()

    return

def plot_two_axes( x, y1, y2, y1label, y2label, filename, xlabel='Time', y1scale='log', y2scale='linear' ):

    fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax1.plot(x, y1, color=color)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label, color=color)
    ax1.set_yscale(y1scale)
    ax1.tick_params(axis='y', which='both', labelcolor=color)

    color = 'tab:orange'
    ax2.plot(x, y2, color=color)
    ax2.set_ylabel(y2label, color=color)
    ax2.set_yscale(y2scale)
    ax2.tick_params(axis='y', which='both', labelcolor=color)

    fig.tight_layout()
    fig.savefig(filename)
    pyplot.close()

    return

def plot_populations( t, x, y, a, xlabel, ylabel, filename, prune=5, xscale='log', xlim=None, ytick=None ):

    fig = pyplot.figure(figsize=(6.4, 4.8), dpi=150)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()

    cmap = cm.viridis(t/t.max())
    for i, yi in enumerate(y.T):
        if i % prune == 0:
            ax1.plot(x, yi, color=cmap[i])

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_xscale(xscale)
    if xlim is not None:
        ax1.set_xlim(*xlim)
    if ytick is not None:
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter(ytick))

    ax2.plot(x, a, 'k--')

    ax2.set_ylabel('Liquid-Phase Partition')
    ax2.set_xscale(xscale)
    if xlim is not None:
        ax2.set_xlim(*xlim)
    ax2.set_ylim(-0.05, 1.05)

    pyplot.colorbar(mappable=cm.ScalarMappable(cmap='viridis', norm=colors.Normalize(vmin=t.min(), vmax=t.max())), location='top', label='Time')

    pyplot.tight_layout()
    pyplot.savefig(filename)
    pyplot.close()

    return

