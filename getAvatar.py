from flask import Flask, request, send_file, render_template_string
from PIL import Image
import torch
import io
import ssl
from flask_cors import CORS  # Import CORS
import logging


logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
app.logger.debug("Processing image...")


# Enable CORS for the entire app
CORS(app)  # This will allow all origins to access your API

# Disable SSL verification for testing
ssl._create_default_https_context = ssl._create_unverified_context

# Load the AnimeGANv2 model and the face2paint pipeline
try:
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
    face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)
    print("Model loaded successfully.")
    status = "1."
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    face2paint = None
    status = f"Error loading model: {e}"

@app.route('/')
def home():
    print("Serving home page...")
    # Display the status message along with the form
    html_form = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AnimeGAN2 Image Upload</title>
    </head>
    <body>
        <h1>Upload Your Image to AnimeGAN2</h1>
        <p>Status: {status}</p>  <!-- Display status here -->
        <form action="/upload" method="POST" enctype="multipart/form-data">
            <label for="image">Choose an image:</label>
            <input type="file" id="image" name="image" accept="image/*" required>
            <br><br>
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    '''
    return render_template_string(html_form)

@app.route('/upload', methods=['POST'])
def upload_image():
    if not model or not face2paint:
        global status
        status = "Error: Model not loaded properly"
        return f"Error: Model not loaded properly", 500

    try:

        print("Processing image...")
        status = "Processing image..."
        # Get the uploaded file
        file = request.files['image']
        img = Image.open(file.stream).convert("RGB")
        print(f"Image uploaded: {file.filename}")
        status = "Processing image..."

        img = img.resize((256, 256))  # Resize the image to a smaller size

        # Process the image using face2paint
        out_img = face2paint(model, img)
        print("Image processed successfully.")
        status = "Image processed successfully."

        # Save or send the processed image
        img_io = io.BytesIO()
        out_img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        print(f"Error during image processing: {e}")
        status = f"Error processing image: {e}"
        return f"Error processing image: {e}", 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
