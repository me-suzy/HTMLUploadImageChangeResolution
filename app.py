# app.py
import os
import json
import zipfile
import tempfile
import shutil
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageEnhance
import io
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Predefined dimensions for devices and social media
DEVICE_DIMENSIONS = {
    'email': (600, 400),
    'desktop': (1920, 1080),
    'tablet': (1024, 768),
    'mobile': (375, 667)
}

SOCIAL_DIMENSIONS = {
    'instagram-post': (1080, 1080),
    'instagram-story': (1080, 1920),
    'facebook-post': (1200, 630),
    'facebook-story': (1080, 1920),
    'twitter-post': (1024, 512),
    'pinterest-pin': (1000, 1500),
    'linkedin-post': (1200, 627)
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def convert_units_to_pixels(value, unit, dpi=72):
    """Convert different units to pixels"""
    try:
        value = float(value)
        if unit == 'px':
            return int(value)
        elif unit == 'cm':
            return int(value * dpi / 2.54)
        elif unit == 'mm':
            return int(value * dpi / 25.4)
        elif unit == 'in':
            return int(value * dpi)
        elif unit == 'pt':
            return int(value * dpi / 72)
        elif unit == 'pc':
            return int(value * dpi / 6)
        else:
            return int(value)
    except:
        return None

def compress_image(image, quality=70):
    """Compress image to reduce file size"""
    try:
        # Convert RGBA to RGB if necessary for JPEG compression
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[-1])
            else:
                background.paste(image)
            image = background
        
        # Create output buffer
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=quality, optimize=True)
        output.seek(0)
        
        # Return compressed image
        return Image.open(output)
    except Exception as e:
        print(f"Error compressing image: {e}")
        return image

def upscale_image_ai(image, scale_factor=2):
    """Upscale image using AI-like techniques"""
    try:
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Calculate new dimensions
        new_width = int(cv_image.shape[1] * scale_factor)
        new_height = int(cv_image.shape[0] * scale_factor)
        
        # Use INTER_CUBIC for better quality upscaling
        upscaled = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply sharpening filter for better quality
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        
        # Convert back to PIL
        upscaled_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
        
        return upscaled_image
    except Exception as e:
        print(f"Error in AI upscaling: {e}")
        # Fallback to simple resize
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        return image.resize(new_size, Image.Resampling.LANCZOS)

def resize_image_by_dimensions(image, width=None, height=None, unit='px', dpi=72):
    """Resize image by specific dimensions"""
    try:
        if width:
            width = convert_units_to_pixels(width, unit, dpi)
        if height:
            height = convert_units_to_pixels(height, unit, dpi)
        
        if not width and not height:
            return image
            
        original_width, original_height = image.size
        
        if width and height:
            # Both dimensions specified - resize exactly
            return image.resize((width, height), Image.Resampling.LANCZOS)
        elif width:
            # Only width specified - maintain aspect ratio
            ratio = width / original_width
            new_height = int(original_height * ratio)
            return image.resize((width, new_height), Image.Resampling.LANCZOS)
        elif height:
            # Only height specified - maintain aspect ratio
            ratio = height / original_height
            new_width = int(original_width * ratio)
            return image.resize((new_width, height), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error resizing by dimensions: {e}")
    
    return image

def resize_image_by_percentage(image, percentage):
    """Resize image by percentage"""
    try:
        percentage = float(percentage) / 100
        new_width = int(image.width * percentage)
        new_height = int(image.height * percentage)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error resizing by percentage: {e}")
        return image

def resize_image_by_aspect_ratio(image, aspect_ratio):
    """Resize image to specific aspect ratio"""
    if not aspect_ratio:
        return image
    
    try:
        width_ratio, height_ratio = map(int, aspect_ratio.split(':'))
        current_width, current_height = image.size
        
        # Calculate new dimensions maintaining the aspect ratio
        if current_width / current_height > width_ratio / height_ratio:
            # Image is wider than target ratio
            new_height = current_height
            new_width = int(new_height * width_ratio / height_ratio)
        else:
            # Image is taller than target ratio
            new_width = current_width
            new_height = int(new_width * height_ratio / width_ratio)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error resizing by aspect ratio: {e}")
        return image

def resize_image_for_device(image, device):
    """Resize image for specific device"""
    if device in DEVICE_DIMENSIONS:
        target_width, target_height = DEVICE_DIMENSIONS[device]
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return image

def resize_image_for_social(image, social):
    """Resize image for social media platform"""
    if social in SOCIAL_DIMENSIONS:
        target_width, target_height = SOCIAL_DIMENSIONS[social]
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return image

def process_single_image(file_path, settings):
    """Process a single image according to settings"""
    try:
        # Open image
        image = Image.open(file_path)
        
        # Apply upscaling first if requested
        if settings.get('upscale', False):
            print("Applying AI upscaling...")
            image = upscale_image_ai(image, scale_factor=2)
        
        # Apply resizing based on type
        resize_type = settings.get('resizeType', 'percentage')
        print(f"Resize type: {resize_type}")
        
        if resize_type == 'dimensions':
            width = settings.get('width')
            height = settings.get('height')
            unit = settings.get('unit', 'px')
            dpi = int(settings.get('dpi', 72)) if settings.get('dpi') else 72
            
            if width or height:
                image = resize_image_by_dimensions(
                    image, 
                    int(width) if width else None,
                    int(height) if height else None,
                    unit, 
                    dpi
                )
        
        elif resize_type == 'percentage':
            percentage = settings.get('percentage', 100)
            image = resize_image_by_percentage(image, percentage)
        
        elif resize_type == 'aspect-ratio':
            aspect_ratio = settings.get('aspectRatio')
            image = resize_image_by_aspect_ratio(image, aspect_ratio)
        
        elif resize_type == 'devices':
            device = settings.get('device')
            image = resize_image_for_device(image, device)
        
        elif resize_type == 'social':
            social = settings.get('social')
            image = resize_image_for_social(image, social)
        
        # Apply compression if requested
        if settings.get('compress', False):
            print("Applying compression...")
            image = compress_image(image, quality=70)
        
        return image
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/process_images', methods=['POST'])
def process_images():
    """Main endpoint to process uploaded images"""
    try:
        print("Starting image processing...")
        
        # Get settings from form data
        settings_json = request.form.get('settings')
        if not settings_json:
            return jsonify({'error': 'No settings provided'}), 400
        
        settings = json.loads(settings_json)
        print(f"Settings: {settings}")
        
        # Get uploaded files
        files = []
        for key in request.files:
            if key.startswith('file_'):
                files.append(request.files[key])
        
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        print(f"Processing {len(files)} files...")
        
        # Create a permanent ZIP file in the processed folder
        zip_filename = 'processed_images.zip'
        zip_path = os.path.join(PROCESSED_FOLDER, zip_filename)
        
        # Remove existing zip if it exists
        if os.path.exists(zip_path):
            os.remove(zip_path)
        
        processed_files = []
        
        # Process each file
        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                print(f"Processing file {i+1}: {file.filename}")
                
                # Save original file
                filename = secure_filename(file.filename)
                original_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(original_path)
                
                # Process image
                processed_image = process_single_image(original_path, settings)
                
                if processed_image:
                    # Save processed image
                    name, ext = os.path.splitext(filename)
                    target_format = settings.get('targetFormat', 'jpg').lower()
                    
                    # Map format names
                    format_map = {
                        'jpg': 'JPEG',
                        'jpeg': 'JPEG',
                        'png': 'PNG',
                        'webp': 'WEBP',
                        'gif': 'GIF',
                        'bmp': 'BMP',
                        'tiff': 'TIFF'
                    }
                    
                    pil_format = format_map.get(target_format, 'JPEG')
                    processed_filename = f"{name}_processed.{target_format}"
                    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
                    
                    # Handle different formats
                    if pil_format == 'JPEG' and processed_image.mode in ('RGBA', 'LA'):
                        # Convert to RGB for JPEG
                        background = Image.new('RGB', processed_image.size, (255, 255, 255))
                        if processed_image.mode == 'RGBA':
                            background.paste(processed_image, mask=processed_image.split()[-1])
                        else:
                            background.paste(processed_image)
                        processed_image = background
                    
                    # Save the processed image
                    processed_image.save(processed_path, format=pil_format, quality=95)
                    processed_files.append(processed_path)
                    print(f"Saved: {processed_filename}")
        
        # Create ZIP file with all processed images
        if processed_files:
            print("Creating ZIP file...")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in processed_files:
                    zipf.write(file_path, os.path.basename(file_path))
            
            print(f"ZIP file created: {zip_path}")
            
            # Send the ZIP file
            return send_file(
                zip_path, 
                as_attachment=True, 
                download_name='processed_images.zip',
                mimetype='application/zip'
            )
        else:
            return jsonify({'error': 'No images could be processed'}), 400
    
    except Exception as e:
        print(f"Error in process_images: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_file('index.html')

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Image Resizer Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Processed folder: {PROCESSED_FOLDER}")
    app.run(debug=True, host='0.0.0.0', port=5000)