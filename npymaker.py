from PIL import Image
import numpy as np

def npymaker(folder_path):
    
    # Load the two images
    image1 = Image.open('image1.jpg')
    image2 = Image.open('image2.jpg')

    # Resize the images if needed
    # image1 = image1.resize((width, height))
    # image2 = image2.resize((width, height))

    # Combine the images horizontally
    combined_image = np.hstack((image1, image2))

    # Convert the combined image to a NumPy array
    numpy_array = np.array(combined_image)

    # Save the NumPy array as a file
    np.save('combined_image.npy', numpy_array)

basic_folder = 'C:\Workspace\CarbonCapturePredict\Dataset\Training\image\SN10_Forest_IMAGE'
npymaker(basic_folder)