# Eigen-Faces-Hough-line-elipse-App

#### Project Overview
This project implements two main functionalities: Eigenfaces Analysis and Hough Analysis, within a graphical user interface (GUI) developed using PyQt5. Eigenfaces Analysis involves analyzing facial images to extract principal components, while Hough Analysis focuses on detecting ellipses in images using the Hough transform technique. The application provides a user-friendly interface for performing these analyses and visualizing the results.

#### Dependencies
- Python 3.x
- PyQt5
- NumPy
- Matplotlib
- OpenCV (cv2)

#### Installation
1. Ensure you have Python 3.x installed on your system.
2. Install the required dependencies using pip:
    ```
    pip install PyQt5 numpy matplotlib opencv-python
    ```
3. Clone or download the project repository to your local machine.

#### Usage
1. Navigate to the project directory.
2. Run the `main.py` script:
    ```
    python main.py
    ```
3. The application window will open, presenting two tabs: "Eigenfaces Analysis" and "Hough Analysis."
4. **Eigenfaces Analysis:**
   - Enter the desired number of components and variance thresholds in the input fields provided.
   - Click the "Plot Eigenfaces" button to visualize the eigenfaces and reconstructed images based on the input parameters.
5. **Hough Analysis:**
   - Input the parameters for Hough ellipse detection, such as `a_min`, `a_max`, `b_min`, `b_max`, etc.
   - Click the "Detect Ellipses" button to perform Hough ellipse detection on the selected image.
   - Use the "Browse Image" button to select an image for analysis.
   - Detected ellipses will be overlaid on the original image, and the result will be displayed in the GUI.

#### Notes
- Ensure that you have the required image dataset (`olivetti_faces.npy`) for Eigenfaces Analysis.
- Experiment with different parameter values for Hough Analysis to optimize ellipse detection.
- This project serves as a basic demonstration of image analysis techniques and can be extended with additional functionalities or optimizations as needed.

#### Contributors
- Hassan Elsheikh
- Ammar Yasser
- Asmaa Khalid
- Nada Alfowey
