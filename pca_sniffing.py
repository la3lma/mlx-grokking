import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def plot_2d(pca_result, dim1, dim2):
    # Plot the PCA result (2D plot of the evolution)
    # Create the colormap: darker at the start, lighter at the end
    colors = cm.cividis(np.linspace(0, 1, len(pca_result)))
    plt.figure(figsize=(8, 6))
    # plt.plot(pca_result[:, 0], pca_result[:, 1], marker='o', linestyle='-', markersize=4)
    for i in range(len(pca_result) - 1):
        plt.plot(pca_result[i:i + 2, dim1 -1 ], pca_result[i:i + 2, dim2 - 1 ], color=colors[i], marker='o', markersize=4)
    plt.title(f'2D PCA Evolution Plot, dimensions {dim1} and {dim2}')
    plt.xlabel(f'Principal Component {dim1}')
    plt.ylabel(f'Principal Component {dim2}')
    plt.grid(True)


    plotname = "pca_" + str(dim1) + "_" + str(dim2) + ".png"
    filename = "plots/" + plotname
    plt.savefig(filename)
    plt.show()


def plot_3d(pca_result):
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
    ax.set_title('3D PCA Evolution Plot with Color Gradient')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    plotname = "pca_3d_1_2_3.png"
    filename = "plots/" + plotname
    plt.savefig(filename)

    # Show the plot
    plt.show()
    plt.show()


def analyze_by_pca(array_data):
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
    # Plot the PCA result
    for i in range(1, n_components):
        for j in range(i + 1, n_components):
            plot_2d(pca_result, i, j)
    plot_3d(pca_result)


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


if __name__ == '__main__':

    # Load the .npz file
    data = np.load('consolidated_weights.npz')

    # List all arrays in the .npz file
    print(data.files)  # This will print all the keys (array names) in the .npz file

    # Access an array by its key
    array_name = 'arr_0'  # Replace with the actual array name
    array_data = data[array_name]
    # Don't forget to close the file after loading
    data.close()

    # analyze_by_pca(array_data)

    # Apply t-SNE to project the data into 2D space
    analyze_t_sne(array_data)


