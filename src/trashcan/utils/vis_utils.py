import matplotlib.pyplot as plt


def plot_pair(images, gray=False):
    fig, axes = plt.subplots(
        nrows=1, ncols=2, sharex=False, sharey=False, figsize=(10, 8)
    )
    i = 0

    for y in range(2):
        if gray:
            axes[y].imshow(images[i], cmap="gray")
        else:
            axes[y].imshow(images[i])
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i += 1

    plt.show()
