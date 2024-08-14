# Face AI Kit UI - Face Recognition

## Overview

The **Face Recognition Comparison Application** is a GUI-based tool that allows users to compare face images using different face recognition models. The application supports selecting a reference image and multiple comparison images, and then computes L2 similarity scores between the faces in these images. The results can be exported in various formats, including PDF and CSV, along with face embeddings.

## Features

- **Multiple Face Recognition Models**: Supports different models like `magface_cwh` and `arcface`.
- **Image Resizing**: Option to resize images to a specified maximum width before processing.
- **Progress Tracking**: Displays progress and logs inference time for each image.
- **Export Options**: Export results and embeddings to PDF or CSV.


## Requirements

- Python 3.6 or higher
- Required Python packages:
  - PyQt5
  - OpenCV (cv2)
  - pandas
  - fpdf
  - face_ai_kit

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/tgoldmann/face-ai-kit-ui
   cd face-ai-kit-ui
   ```

2. **Install the required packages:**

   You can install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   If you have a custom face recognition library, make sure it is installed and accessible in your Python environment.

3. **Run the application:**

   ```bash
   python main.py
   ```

## Usage

1. **Start the Application**: Run the application using the command above.

2. **Select Face Recognition Model**: Choose a model from the dropdown menu.

3. **Set Confidence Threshold**: Input the desired confidence threshold for face comparison.

4. **Select Images**:
   - **Reference Image**: Click "Select Reference Image" and choose an image file.
   - **Comparison Images**: Click "Select Comparison Images" and choose one or more images for comparison.

5. **Resize Images (Optional)**:
   - Check the "Resize images to max width" checkbox if you want to resize images before processing.
   - Enter the maximum width in pixels in the input field next to the checkbox.

6. **Compare Faces**: Click the "Compare" button to start the face comparison process. The progress window will show the progress and log details, including the inference time for each image.

7. **View Results**: The results will be displayed in the results section of the application.

8. **Export Results**:
   - **PDF**: Click "Export PDF" to save the comparison results in PDF format.
   - **CSV**: Click "Export CSV" to save the results in CSV format.
   - **Embeddings**: Click "Export Embeddings" to save face embeddings in CSV format.



## Contributing

If you want to contribute to this project, feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Please note that the face recognition models (magface_cwh, arcface, etc.) used in this application are not intended for commercial purposes. Ensure you adhere to the licensing terms of these models when using this application.


## Acknowledgments

- [PyQt5](https://pypi.org/project/PyQt5/) for the GUI framework.
- [OpenCV](https://opencv.org/) for image processing.
- [pandas](https://pandas.pydata.org/) for data handling.
- [FPDF](http://www.fpdf.org/) for PDF generation.
```

