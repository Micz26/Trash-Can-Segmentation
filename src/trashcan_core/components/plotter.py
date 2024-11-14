import matplotlib.pyplot as plt


class Plotter:
    def plot_pair(self, images, gray=False):
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

    def plot_predicted_channels(self, im):
        # Detach the tensor from the computation graph and convert it to numpy
        im_np = im.detach().cpu().numpy()

        # Get number of channels (e.g., 17 channels)
        n_channels = im_np.shape[0]
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
        fig.suptitle("Channels of the First Image in Batch")

        for i in range(n_channels):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]

            # Make sure to handle the 2D shape of each channel (height, width)
            channel_image = im_np[
                i
            ]  # This should have shape [256, 480] (height, width)

            # Check shape of the channel before plotting
            if channel_image.ndim == 2:  # 2D array (height, width)
                ax.imshow(channel_image, cmap="viridis")
            else:
                ax.imshow(channel_image.squeeze(), cmap="viridis")  # Ensure it is 2D
            ax.set_title(f"Channel {i + 1}")
            ax.axis("off")

        for j in range(i + 1, n_rows * n_cols):
            row, col = divmod(j, n_cols)
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_segmentation(self, im):
        im_np = im.cpu().numpy()

        plt.imshow(im_np, cmap="viridis")
        plt.colorbar()
        plt.title("Visualization of First Image Prediction")
        plt.show()
