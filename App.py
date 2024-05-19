from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QTabWidget, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas



# load the dataset ( a numpy array ) 
def load_olivetti_faces(dataset_path):
    images = np.load(dataset_path)
    return images



# resize and flatten the image
def preprocess_images(images, target_size=(64, 64)):
    resized_images = [cv2.resize(img, target_size) for img in images]
    flattened_images = [img.flatten() for img in resized_images]
    return np.array(flattened_images)

# averaging along the rows of the images array
def calculate_mean_face(images):
    return np.mean(images, axis=0)

# mean_face a 1D array representing the average intensity values for each pixel position across all images
def calculate_covariance_matrix(images):
    mean_face = calculate_mean_face(images)
    centered_images = images - mean_face
    # transposes the centered_images array
    covariance_matrix = np.cov(centered_images.T)
    return covariance_matrix

# perform eigen decomposition on the covariance matrix and sort the eigenvalues and eigenvectors
def perform_eigen_decomposition(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors

# select the top num_components eigenvectors, which are the eigenfaces
def compute_eigenfaces(eigenvectors, num_components):
    return eigenvectors[:, :num_components]

# project the images onto the eigenfaces
def project_images(images, eigenfaces):
    return np.dot(images, eigenfaces)

# reconstruct the images from their projections onto the eigenfaces
def reconstruct_images(projection, eigenfaces, mean_face):
    return np.dot(projection, eigenfaces.T) + mean_face

def components_for_variance(eigenvalues, variance_threshold):
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    return num_components



#Hough
def detect_and_draw_hough_ellipses(image_path, a_min=19, a_max=41, b_min=19, b_max=41, delta_a=14, delta_b=14, num_thetas=100, bin_threshold=0.195, min_edge_threshold=50, max_edge_threshold=150, sampling_ratio=0.59):
    """
    
        a_min (int): Minimum semi-major axis length of ellipses to detect.
        a_max (int): Maximum semi-major axis length of ellipses to detect.
        b_min (int): Minimum semi-minor axis length of ellipses to detect.
        b_max (int): Maximum semi-minor axis length of ellipses to detect.
        delta_a (int): Step size for semi-major axis length.
        delta_b (int): Step size for semi-minor axis length.
        num_thetas (int): Number of steps for theta from 0 to 2PI.
        bin_threshold (float): Thresholding value in percentage to shortlist candidate ellipses.
        min_edge_threshold (int): Minimum threshold value for edge detection.
        max_edge_threshold (int): Maximum threshold value for edge detection.
        sampling_ratio (float): Ratio of edge points to sample for ellipse detection.
    
    """

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edge_image = cv2.Canny(gray_image, min_edge_threshold, max_edge_threshold)

    # Get image dimensions
    img_height, img_width = edge_image.shape[:2]

    # Initialize parameters for Hough ellipse detection
    dtheta = int(360 / num_thetas)
    thetas = np.arange(0, 360, step=dtheta)
    as_ = np.arange(a_min, a_max, step=delta_a)
    bs = np.arange(b_min, b_max, step=delta_b)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    num_thetas = int(num_thetas)
    print("num_thetas", num_thetas)
    ellipse_candidates = [(a, b, int(a * cos_thetas[t]), int(b * sin_thetas[t]))
                            for a in as_ for b in bs for t in range(num_thetas)]

    # Initialize accumulator
    accumulator = defaultdict(int)

    # Get edge points
    edge_points = [(x, y) for y in range(img_height) for x in range(img_width) if edge_image[y, x] != 0]

    print(f"Total edge points: {len(edge_points)}")

    # Sample a subset of edge points
    if len(edge_points) > 0:
        sampled_edge_points = random.sample(edge_points, int(len(edge_points) * sampling_ratio))
    else:
        sampled_edge_points = edge_points

    print(f"Sampled edge points: {len(sampled_edge_points)}")

    # Iterate over sampled edge points and vote for potential ellipse centers
    for x, y in sampled_edge_points:
        for a, b, acos_t, bsin_t in ellipse_candidates:
            x_center = x - acos_t
            y_center = y - bsin_t
            accumulator[(x_center, y_center, a, b)] += 1

    print(f"Total votes in accumulator: {len(accumulator)}")

    # Initialize output image
    output_img = image.copy()

    # Store detected ellipses
    out_ellipses = []

    # Loop through the accumulator to find ellipses based on the threshold
    for candidate_ellipse, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y, a, b = candidate_ellipse
        current_vote_percentage = votes / num_thetas
        if current_vote_percentage >= bin_threshold:
            out_ellipses.append((x, y, a, b, current_vote_percentage))

    print(f"Detected ellipses before post-processing: {len(out_ellipses)}")

    # Perform post-processing to remove duplicate ellipses
    pixel_threshold = 10
    postprocess_ellipses = []
    for x, y, a, b, v in out_ellipses:
        if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(a - ac) > pixel_threshold or abs(b - bc) > pixel_threshold for xc, yc, ac, bc, v in postprocess_ellipses):
            postprocess_ellipses.append((x, y, a, b, v))
    out_ellipses = postprocess_ellipses

    print(f"Detected ellipses after post-processing: {len(out_ellipses)}")

    return output_img, out_ellipses


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Eigenfaces and Hough Analysis")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

        # Tab widget for Eigenfaces and Hough Analysis
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Tab for Eigenfaces Analysis
        self.eigen_tab = QWidget()
        self.tab_widget.addTab(self.eigen_tab, "Eigenfaces Analysis")
        self.setup_eigen_tab()

        # Tab for Hough Analysis
        self.hough_tab = QWidget()
        self.tab_widget.addTab(self.hough_tab, "Hough Analysis")
        self.setup_hough_tab()

    def setup_eigen_tab(self):
        layout = QVBoxLayout()
        self.eigen_tab.setLayout(layout)

        # Input fields for number of components and variance thresholds
        self.components_input = QLineEdit()
        self.variance_input = QLineEdit()
        layout.addWidget(QLabel("Number of Components (comma-separated):"))
        layout.addWidget(self.components_input)
        layout.addWidget(QLabel("Variance Thresholds (comma-separated):"))
        layout.addWidget(self.variance_input)

        # Button to trigger plotting for Eigenfaces
        self.plot_button_eigen = QPushButton("Plot Eigenfaces")
        self.plot_button_eigen.clicked.connect(self.plot_eigen)
        layout.addWidget(self.plot_button_eigen)

        # Plot for Cumulative Variance Explained
        self.fig_components, self.ax_components = plt.subplots()
        self.canvas_components = FigureCanvas(self.fig_components)
        layout.addWidget(self.canvas_components)

        # Plot for Reconstructed Images
        self.fig_images, self.ax_images = plt.subplots()
        self.canvas_images = FigureCanvas(self.fig_images)
        layout.addWidget(self.canvas_images)

    def setup_hough_tab(self):
        layout = QVBoxLayout()
        self.hough_tab.setLayout(layout)

        # Input fields for Hough Analysis parameters
        self.a_min_input = QLineEdit()
        self.a_max_input = QLineEdit()
        self.b_min_input = QLineEdit()
        self.b_max_input = QLineEdit()
        self.delta_a_input = QLineEdit()
        self.delta_b_input = QLineEdit()
        self.bin_threshold_input = QLineEdit()
        self.sampling_ratio_input = QLineEdit()

        layout.addWidget(QLabel("Hough Analysis Parameters:"))
        layout.addWidget(QLabel("a_min (int):"))
        layout.addWidget(self.a_min_input)
        layout.addWidget(QLabel("a_max (int):"))
        layout.addWidget(self.a_max_input)
        layout.addWidget(QLabel("b_min (int):"))
        layout.addWidget(self.b_min_input)
        layout.addWidget(QLabel("b_max (int):"))
        layout.addWidget(self.b_max_input)
        layout.addWidget(QLabel("delta_a (int):"))
        layout.addWidget(self.delta_a_input)
        layout.addWidget(QLabel("delta_b (int):"))
        layout.addWidget(self.delta_b_input)
        layout.addWidget(QLabel("bin_threshold (float):"))
        layout.addWidget(self.bin_threshold_input)
        layout.addWidget(QLabel("sampling_ratio (float):"))
        layout.addWidget(self.sampling_ratio_input)

        # Button to trigger Hough analysis
        self.plot_button_hough = QPushButton("Detect Ellipses")
        self.plot_button_hough.clicked.connect(self.plot_hough)
        layout.addWidget(self.plot_button_hough)

        #Add Browse Image Button
        self.browse_button = QPushButton("Browse Image")
        self.browse_button.clicked.connect(self.browse_image)
        layout.addWidget(self.browse_button)

        from PyQt5.QtWidgets import QHBoxLayout

        # Create a horizontal layout for the plots
        plot_layout = QHBoxLayout()

        # Plot for original Image
        self.fig_original, self.ax_original = plt.subplots(figsize=(10, 10), frameon=True)  # Adjust width and height as needed
        self.canvas_original = FigureCanvas(self.fig_original)
        plot_layout.addWidget(self.canvas_original)
        self.ax_original.axis('off')

        # Plot for Hough Ellipses
        self.fig_hough, self.ax_hough = plt.subplots(figsize=(10, 10), frameon=True)  # Adjust width and height as needed
        self.canvas_hough = FigureCanvas(self.fig_hough)
        plot_layout.addWidget(self.canvas_hough)
        self.ax_hough.axis('off')
        # Add the plot layout to the main layout
        layout.addLayout(plot_layout)


    def plot_eigen(self):
        # Retrieve user input
        components = list(map(int, self.components_input.text().split(',')))
        variances = list(map(float, self.variance_input.text().split(',')))


        # Load dataset and perform analysis
        dataset_path = "olivetti_faces.npy"
        images = load_olivetti_faces(dataset_path)
        preprocessed_images = preprocess_images(images)
        mean_face = calculate_mean_face(preprocessed_images)
        covariance_matrix = calculate_covariance_matrix(preprocessed_images)
        eigenvalues, eigenvectors = perform_eigen_decomposition(covariance_matrix)

        components_for_variances = [components_for_variance(eigenvalues, variance) for variance in variances]

        # Calculate cumulative variance explained
        cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

        # Plot cumulative variance explained
        self.ax_components.clear()
        self.ax_components.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
        self.ax_components.set_xlabel('Number of Components')
        self.ax_components.set_ylabel('Cumulative Variance Explained')
        self.ax_components.set_title('Cumulative Variance Explained by Principal Components')
        self.ax_components.grid(True)
        self.canvas_components.draw()

        #Plot images based on the number of components
        num_images = 5
        selected_indices = random.sample(range(len(preprocessed_images)), num_images)
        selected_images = preprocessed_images[selected_indices]
        # Visualize reconstructed images based on component list
        fig, axes = plt.subplots(num_images, len(components) + 1, figsize=(15, 10))
        for i in range(num_images):
            axes[i, 0].imshow(selected_images[i].reshape(64, 64), cmap='gray')
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            for j, num_components in enumerate(components):
                eigenfaces = compute_eigenfaces(eigenvectors, num_components)
                centered_image = selected_images[i] - mean_face
                projection = project_images(np.array([centered_image]), eigenfaces)
                reconstructed_image = reconstruct_images(projection, eigenfaces, mean_face)
                axes[i, j+1].imshow(reconstructed_image.reshape(64, 64), cmap='gray')
                axes[i, j+1].set_title(f'{num_components} Components')
                axes[i, j+1].axis('off')
        plt.tight_layout()
        plt.show()

        ## Visualize reconstructed images based on variance thresholds
        fig, axes = plt.subplots(num_images, len(variances) + 1, figsize=(20, 10))
        for i in range(num_images):
            axes[i, 0].imshow(selected_images[i].reshape(64, 64), cmap='gray')
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            for j, (num_components, variance_threshold) in enumerate(zip(components_for_variances, variances)):
                eigenfaces = compute_eigenfaces(eigenvectors, num_components)
                centered_image = selected_images[i] - mean_face
                projection = project_images(np.array([centered_image]), eigenfaces)
                reconstructed_image = reconstruct_images(projection, eigenfaces, mean_face)
                axes[i, j + 1].imshow(reconstructed_image.reshape(64, 64), cmap='gray')
                axes[i, j + 1].set_title(f'{int(variance_threshold * 100)}% Variance\n({num_components} Components)')
                axes[i, j + 1].axis('off')
        plt.tight_layout()
        plt.show()

    def browse_image(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open Image', 'C:\\', 'Image Files (*.jpg *.png)')
        self.image_path = file_name[0]
        image = cv2.imread(self.image_path)
        self.ax_original.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.canvas_original.draw()

    def plot_hough(self):
        # Retrieve user input
        a_min = int(self.a_min_input.text())
        a_max = int(self.a_max_input.text())
        b_min = int(self.b_min_input.text())
        b_max = int(self.b_max_input.text())
        delta_a = int(self.delta_a_input.text())
        delta_b = int(self.delta_b_input.text())
        bin_threshold = float(self.bin_threshold_input.text())
        sampling_ratio = float(self.sampling_ratio_input.text())

        # Perform Hough ellipse detection
        output_img, out_ellipses = detect_and_draw_hough_ellipses(self.image_path, a_min, a_max, b_min, b_max, delta_a, delta_b, num_thetas=36, bin_threshold=bin_threshold, min_edge_threshold=50, max_edge_threshold=150, sampling_ratio=sampling_ratio)

    # Draw detected ellipses on the output image
        for x, y, a, b, v in out_ellipses:
            output_img = cv2.ellipse(output_img, (x, y), (a, b), 0, 0, 360, (0, 255, 0), 2)

        # Plot the original image and the output image with detected ellipses
        self.ax_hough.clear()
        self.ax_hough.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        self.ax_hough.axis('off')
        self.canvas_hough.draw()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())