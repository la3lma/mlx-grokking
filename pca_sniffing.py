import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def plot_2d(title, pca_result, dim1, dim2, val_acc_trace=None):
    # Plot the PCA result (2D plot of the evolution)
    # Create the colormap: darker at the start, lighter at the end, or if a val_acc_trace is provided, use it
    # to color the lines, being lighter the closer to 100 % accuracy (index 100 in the colormap).

    val_acc_trace_colors = cm.cividis(np.linspace(0, 1, 101))

    colors = cm.cividis(np.linspace(0, 1, len(pca_result)))

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, dim1 - 1], pca_result[:, dim2 - 1], c=colors, cmap='viridis')
    for i in range(len(pca_result) - 1):
        if val_acc_trace is None:
            plt.plot(pca_result[i:i + 2, dim1 - 1], pca_result[i:i + 2, dim2 - 1], color=colors[i], marker='o', markersize=4)

        else:
            color_index = int(val_acc_trace[i] * 100)
            plt.plot(pca_result[i:i + 2, dim1 -1 ], pca_result[i:i + 2, dim2 - 1 ], color=val_acc_trace_colors[color_index], marker='o', markersize=4)

    plt.title(f'2D PCA Evolution Plot ({title}), dimensions {dim1} and {dim2}')
    plt.xlabel(f'Principal Component {dim1}')
    plt.ylabel(f'Principal Component {dim2}')
    plt.grid(True)

    plotname = "pca_" + str(dim1) + "_" + str(dim2) + ".png"
    filename = "plots/" + plotname


    plt.savefig(filename)
    plt.show()


def plot_3d(title, pca_result):
    # Create the colormap: darker at the start, lighter at the end
    colors = cm.cividis(np.linspace(0, 1, len(pca_result)))
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot each point in 3D space, coloring them according to their order
    for i in range(len(pca_result) - 1):
        ax.plot(pca_result[i:i + 2, 0], pca_result[i:i + 2, 1], pca_result[i:i + 2, 2],
                color=colors[i], marker='o', markersize=4)
    # Add title, labels, and grid
    ax.set_title(f'3D PCA Evolution Plot ({title}) with Color Gradient')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    plotname = "pca_3d_1_2_3.png"
    filename = "plots/" + plotname
    plt.savefig(filename)

    # Show the plot
    plt.show()
    plt.show()


def analyze_by_pca(title, array_data, val_acc_trace=None):
    # Standardize the data (mean=0, variance=1) before applying PCA
    scaler = StandardScaler()
    array_data_scaled = scaler.fit_transform(array_data)
    # Initialize the PCA model, n_components determines how many principal components to keep
    n_components = 10  # For example, reduce to 2 dimensions
    pca = PCA(n_components=n_components)
    # Fit the model and apply the dimensionality reduction
    pca_result = pca.fit_transform(array_data_scaled)
    # Print the result
    # print(f"PCA Result:\n{pca_result}")
    # You can also access the explained variance ratio
    print(f"Explained Variance Ratio:\n{pca.explained_variance_ratio_}")
    plot_explained_variance(pca.explained_variance_ratio_)
    # Plot the PCA result


    for i in range(1, n_components):
        for j in range(i + 1, n_components):
            plot_2d(title, pca_result, i, j, val_acc_trace)
    plot_3d(title, pca_result)

def plot_explained_variance(explained_variance):
    # Plot the explained variance ratio
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(explained_variance))
    plt.title('Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.show()


def analyze_t_sne(array_data):
    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(array_data)

    plot_t_sne(tsne_results, 1, 2)
    plot_t_sne(tsne_results, 1, 3)
    plot_t_sne(tsne_results, 2, 3)


def plot_t_sne(tsne_results, dim1, dim2 ):
    # Plot the t-SNE results
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, dim1-1], tsne_results[:, dim2-1], c=range(len(tsne_results)), cmap='viridis')
    plt.colorbar(label='Index (row number in array)')
    plt.title(f't-SNE visualization of the evolution of the process ({dim1} vs {dim2})')
    plt.xlabel(f't-SNE component {dim1}')
    plt.ylabel(f't-SNE component {dim2}')
    # Optionally, connect the points to visualize the path
    plt.plot(tsne_results[:,  dim1-1], tsne_results[:, dim2-1], color='black', alpha=0.5)
    plt.show()


def print_as_image(array_2d):
    # array_normalized = (array_2d - np.min(array_2d)) / (np.max(array_2d) - np.min(array_2d))
    # Convert normalized array to grayscale (0 to 255)
    # array_grayscale = (array_normalized * 255).astype(np.uint8)
    array_grayscale = array_2d

    reshaped_array = array_grayscale.reshape(150, 384,1433)
    # Set up the figure and axis
    fig, ax = plt.subplots()
    # Set up the initial image
    image = ax.imshow(reshaped_array[0], cmap='gray', vmin=0, vmax=1)

    # Function to update the image for each frame
    def update(frame):
        the_image = reshaped_array[frame]
        image.set_array(the_image)
        ax.set_title(f'Frame {frame + 1}')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(reshaped_array), interval=200)

    # Display the animation
    plt.show()
    ani.save('results/animation.mp4', writer='ffmpeg', fps=5)
    plt.close(fig)


if __name__ == '__main__':

    # Load error data from the epochs
    error_data = np.load('results/errors_and_accuracies.npz')
    train_error_trace = error_data['train_error_trace']
    train_acc_trace = error_data['train_acc_trace']
    val_error_trace = error_data['val_error_trace']
    val_acc_trace = error_data['val_acc_trace']
    error_data.close()

    # Load the .npz file with weights from the learning session (snapshots per epoch)
    data = np.load('results/consolidated_weights.npz')
    # Access an array by its key
    array_name = 'arr_0'  # Replace with the actual array name
    array_data = data[array_name]
    # Don't forget to close the file after loading
    data.close()

    # Print the array data as a picture

    print_as_image(array_data)

    analyze_by_pca("weights", array_data, val_acc_trace)

    # Calculate the difference between consecutive rows
    diff_array = np.diff(array_data, axis=0)
    analyze_by_pca("differential", diff_array)

    # Apply t-SNE to project the data into 2D space
    analyze_t_sne(array_data)




