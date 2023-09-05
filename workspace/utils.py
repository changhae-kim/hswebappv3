from matplotlib import cm, colors, pyplot, ticker


def plot_curves( X, Y, ylabel, filename, xlabel=r'$\tilde{t}$', labels=None, loc='best', yscale='linear', xlim=None, ylim=None, font=16, size=(6.4, 4.2) ):

    pyplot.rcParams.update({'font.size': font})

    pyplot.figure(figsize=size, dpi=150)
    if labels is None:
        for x, y in zip(X, Y):
            pyplot.plot(x, y)
    else:
        for x, y, label in zip(X, Y, labels):
            pyplot.plot(x, y, label=label)
        pyplot.legend(loc=loc)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.yscale(yscale)
    if xlim is not None:
        pyplot.xlim(*xlim)
    if ylim is not None:
        pyplot.ylim(*ylim)
    pyplot.tight_layout()
    pyplot.savefig(filename)
    pyplot.close()

    return

def plot_two_axes( x, y1, y2, y1label, y2label, filename, xlabel=r'$\tilde{t}$', y1scale='log', y2scale='linear', xlim=None, font=16, size=(6.4, 4.2) ):

    pyplot.rcParams.update({'font.size': font})

    fig = pyplot.figure(figsize=size, dpi=150)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax1.plot(x, y1, color=color)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label, color=color)
    ax1.set_yscale(y1scale)
    if xlim is not None:
        ax1.set_xlim(*xlim)
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

def plot_populations( t, x, y, a, xlabel, ylabel, filename, prune=10, tlabel=r'$\tilde{t}$', alabel=r'$1-\alpha$', xscale='log', xlim=None, ytick=None, font=14, size=(6.4, 4.8) ):

    pyplot.rcParams.update({'font.size': font})

    fig = pyplot.figure(figsize=size, dpi=150)
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

    ax2.set_ylabel(alabel)
    ax2.set_xscale(xscale)
    if xlim is not None:
        ax2.set_xlim(*xlim)
    ax2.set_ylim(-0.05, 1.05)

    pyplot.colorbar(mappable=cm.ScalarMappable(cmap='viridis', norm=colors.Normalize(vmin=t.min(), vmax=t.max())), location='top', label=tlabel) ##

    pyplot.tight_layout()
    pyplot.savefig(filename)
    pyplot.close()

    return

def plot_colormap( x, y, z, xlabel, ylabel, zlabel, filename, xscale='linear', yscale='linear', zscale='linear', zmin=None, zmax=None, font=16, size=(6.4, 4.8) ):

    pyplot.rcParams.update({'font.size': font})

    pyplot.figure(figsize=size, dpi=150)
    if zscale == 'log':
        norm = colors.LogNorm(vmin=zmin, vmax=zmax)
    else:
        norm = colors.Normalize(vmin=zmin, vmax=zmax)
    pyplot.pcolormesh(x, y, z, norm=norm)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.xscale(xscale)
    pyplot.yscale(yscale)
    pyplot.colorbar(label=zlabel)
    pyplot.tight_layout()
    pyplot.savefig(filename)
    pyplot.close()

    return

