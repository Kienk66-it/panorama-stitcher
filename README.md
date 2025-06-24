# Panorama Stitcher  

## Overview  

Panorama Stitcher is a Python application for seamlessly stitching multiple images into a single panoramic image. The application offers both a graphical user interface (GUI) built with Tkinter and a command-line interface (CLI). It utilizes advanced computer vision techniques from OpenCV, including SIFT feature detection, homography computation (RANSAC or linear), and blending methods (simple, feather, or multiband). Users can stitch images from files or capture them directly from a camera.  

### Key Features:  
- User-friendly GUI for image selection, camera capture, and result preview  
- CLI support for batch processing  
- Automatic black border cropping for clean panoramas  
- Multiple blending and homography methods for optimal results  

## Installation  

### System Requirements  
- Python 3.7 or higher  
- OS: Windows, Linux, or macOS  
- Compatible camera (for live capture mode)  
- (Optional) GStreamer on Linux for improved camera performance  

### Library Installation  

```bash  
# Clone the repository  
git clone https://github.com/Kienk66-it/panorama-stitcher.git  
cd panorama-stitcher  

# Install required libraries  
pip install -r requirements.txt  

# (Optional) On Linux, install GStreamer for camera support  
sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-good  
```  

## Usage  

### GUI Mode  

Launch the application with graphical interface:  

```bash  
python panorama_stitcher.py --gui  
```  

* **File Mode**: Select multiple images, configure blending and homography methods, then generate panorama.  
* **Camera Mode**: Connect a camera, capture images, and stitch them into a panorama.  

### CLI Mode  

Stitch images via command line:  

```bash  
python panorama_stitcher.py image1.jpg image2.jpg [output.jpg] [blend_method] [homography_method]  
```  

Example:  

```bash  
python panorama_stitcher.py img1.jpg img2.jpg panorama.jpg feather ransac  
```  

* **Blending Methods**: `simple`, `feather`, `multiband`  
* **Homography Methods**: `linear`, `ransac`  
* **Additional Options**:  
  * `--no-crop`: Disable auto-cropping  
  * `--display`: Show results with Matplotlib  

## Notes  

* Ensure sufficient system memory (8GB RAM recommended) for large panoramas.  
* Temporary files are stored in `/tmp/panorama_captures` (Linux) or system temp directory.  
* On Windows, the camera uses OpenCV backend instead of GStreamer.  

## Contact  

For questions or support:  
* Open an issue on GitHub  
* Email: [lekien66.it@gmail.com](mailto:lekien66.it@gmail.com)  
