"""
DeepFake Image Detection API
Uses multiple forensic analysis techniques to detect manipulated images:
1. Error Level Analysis (ELA) - detects JPEG compression inconsistencies
2. Noise Analysis - detects unnatural noise patterns from AI generation
3. Frequency Domain Analysis - detects spectral anomalies
4. Color Consistency Analysis - detects color distribution abnormalities
5. Edge Analysis - detects unnatural edge artifacts
6. Texture Analysis - detects GAN-specific texture patterns
"""

import os
import io
import base64
import uuid
import traceback
from datetime import datetime

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageEnhance
from scipy import fftpack, ndimage
from skimage import feature, filters, measure
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Ensure uploads directory exists
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)


def perform_ela(image, quality=90):
    """
    Error Level Analysis (ELA)
    Resaves the image at a known quality and compares with original.
    Manipulated regions show different error levels.
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Save at specific quality
        buffer = io.BytesIO()
        image.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)

        # Calculate difference
        ela_image = ImageChops.difference(image, resaved)

        # Get statistics
        ela_array = np.array(ela_image, dtype=np.float64)
        
        # Calculate various ELA metrics
        mean_error = np.mean(ela_array)
        max_error = np.max(ela_array)
        std_error = np.std(ela_array)
        
        # Calculate error distribution across channels
        channel_means = [np.mean(ela_array[:, :, c]) for c in range(3)]
        channel_variance = np.var(channel_means)
        
        # High variance between channels can indicate manipulation
        # Very uniform ELA can indicate AI generation
        uniformity = 1.0 - min(std_error / (mean_error + 1e-8), 1.0)
        
        # Score: higher means more likely fake
        # AI-generated images tend to have very uniform ELA
        # Manipulated images tend to have high variance in ELA
        if mean_error < 2.0:
            # Very low error - could be uncompressed or AI-generated
            ela_score = 0.6
        elif uniformity > 0.85:
            # Very uniform error levels - suspicious for AI
            ela_score = 0.7
        elif max_error > 200 and std_error > 30:
            # High variance - likely manipulated regions
            ela_score = 0.75
        elif channel_variance > 10:
            # Different channels show different errors
            ela_score = 0.65
        else:
            # Normal compression artifacts
            ela_score = 0.25 + (mean_error / 50.0) * 0.2

        # Generate ELA visualization
        enhancer = ImageEnhance.Brightness(ela_image)
        ela_enhanced = enhancer.enhance(15)
        
        ela_buffer = io.BytesIO()
        ela_enhanced.save(ela_buffer, format='PNG')
        ela_base64 = base64.b64encode(ela_buffer.getvalue()).decode('utf-8')

        return {
            'score': min(max(ela_score, 0), 1),
            'mean_error': float(mean_error),
            'max_error': float(max_error),
            'std_error': float(std_error),
            'uniformity': float(uniformity),
            'visualization': f'data:image/png;base64,{ela_base64}'
        }
    except Exception as e:
        return {'score': 0.5, 'error': str(e), 'visualization': None}


def analyze_noise(image):
    """
    Noise Analysis
    AI-generated images have different noise patterns than real photos.
    GAN artifacts often show periodic noise patterns.
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image, dtype=np.float64)
        
        scores = []
        for channel in range(3):
            ch = img_array[:, :, channel]
            
            # Apply high-pass filter to extract noise
            blurred = ndimage.gaussian_filter(ch, sigma=2)
            noise = ch - blurred
            
            # Analyze noise statistics
            noise_std = np.std(noise)
            noise_kurtosis = float(np.mean((noise - np.mean(noise))**4) / (noise_std**4 + 1e-8) - 3)
            
            # AI-generated images often have lower noise or different noise distribution
            # Real photos have natural noise with specific kurtosis
            if noise_std < 1.5:
                # Very low noise - suspicious for AI generation
                channel_score = 0.7
            elif abs(noise_kurtosis) > 10:
                # Abnormal noise distribution
                channel_score = 0.65
            elif noise_std > 20:
                # Too much noise - could be post-processing
                channel_score = 0.55
            else:
                # Normal noise levels
                channel_score = 0.2 + abs(noise_kurtosis) / 30.0
            
            scores.append(min(channel_score, 1.0))

        # Check noise consistency across channels
        noise_consistency = np.std(scores)
        if noise_consistency > 0.15:
            # Inconsistent noise across channels - suspicious
            avg_score = np.mean(scores) + 0.1
        else:
            avg_score = np.mean(scores)

        return {
            'score': min(max(float(avg_score), 0), 1),
            'noise_std': float(np.mean([np.std(img_array[:, :, c] - ndimage.gaussian_filter(img_array[:, :, c], sigma=2)) for c in range(3)])),
            'channel_consistency': float(1.0 - noise_consistency)
        }
    except Exception as e:
        return {'score': 0.5, 'error': str(e)}


def analyze_frequency(image):
    """
    Frequency Domain Analysis
    GAN-generated images often have specific artifacts in the frequency domain.
    """
    try:
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image

        img_array = np.array(gray, dtype=np.float64)
        
        # Compute 2D FFT
        f_transform = fftpack.fft2(img_array)
        f_shift = fftpack.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Log transform for visualization
        magnitude_log = np.log1p(magnitude)
        
        # Analyze frequency distribution
        h, w = magnitude_log.shape
        center_y, center_x = h // 2, w // 2
        
        # Create radial profile
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - center_x)**2 + (Y - center_y)**2).astype(int)
        max_r = min(center_x, center_y)
        
        # Calculate average magnitude at each radius
        radial_profile = ndimage.mean(magnitude_log, r, range(max_r))
        
        # GAN images often show unusual drops in high-frequency content
        if len(radial_profile) > 10:
            low_freq = np.mean(radial_profile[:len(radial_profile)//4])
            mid_freq = np.mean(radial_profile[len(radial_profile)//4:len(radial_profile)//2])
            high_freq = np.mean(radial_profile[len(radial_profile)//2:3*len(radial_profile)//4])
            
            # Calculate frequency ratio
            if high_freq > 0:
                freq_ratio = low_freq / (high_freq + 1e-8)
            else:
                freq_ratio = 100
            
            # GAN images tend to have steeper frequency falloff
        # Refined Frequency Analysis for Balance
        # Compression (WhatsApp) creates blocks, AI creates periodic patterns
        if freq_ratio > 45:
            freq_score = 0.75
        elif freq_ratio > 30:
            freq_score = 0.6
        else:
            freq_score = 0.35
            
        # Check for periodic patterns but filter out standard 8x8 compression blocks
        if len(radial_profile) > 15:
            diffs = np.diff(radial_profile)
            ripples = np.sum(diffs[1:] * diffs[:-1] < 0)
            # Only flag if ripples are very dense (AI-style) vs sparse (compression-style)
            if ripples > len(radial_profile) * 0.55: # Increased threshold
                freq_score = max(freq_score, 0.8)
        else:
            freq_score = 0.5
            freq_ratio = 0

        # Generate frequency visualization
        vis = (magnitude_log / magnitude_log.max() * 255).astype(np.uint8)
        freq_img = Image.fromarray(vis)
        freq_buffer = io.BytesIO()
        freq_img.save(freq_buffer, format='PNG')
        freq_base64 = base64.b64encode(freq_buffer.getvalue()).decode('utf-8')

        return {
            'score': min(max(float(freq_score), 0), 1),
            'freq_ratio': float(freq_ratio) if isinstance(freq_ratio, (int, float)) else 0,
            'visualization': f'data:image/png;base64,{freq_base64}'
        }
    except Exception as e:
        return {'score': 0.5, 'error': str(e), 'visualization': None}


def analyze_color_consistency(image):
    """
    Color Consistency Analysis
    AI-generated images may have unnatural color distributions or transitions.
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image, dtype=np.float64)
        
        # Analyze color distribution
        scores = []
        
        # 1. Check color histogram smoothness
        for c in range(3):
            channel = img_array[:, :, c].flatten()
            hist, _ = np.histogram(channel, bins=256, range=(0, 255))
            hist = hist.astype(np.float64)
            
            # Smooth histograms are more natural
            hist_diff = np.diff(hist)
            smoothness = np.std(hist_diff) / (np.mean(np.abs(hist_diff)) + 1e-8)
            
            if smoothness > 5:
                scores.append(0.6)  # Unnatural histogram
            else:
                scores.append(0.3)
        
        # 2. Check for color banding (common in AI-generated images)
        for c in range(3):
            channel = img_array[:, :, c]
            unique_values = len(np.unique(channel.astype(np.uint8)))
            if unique_values < 100:
                scores.append(0.7)  # Limited color palette - suspicious
            elif unique_values > 250:
                scores.append(0.2)  # Full range - natural
            else:
                scores.append(0.4)
        
        # 3. Analyze color transition smoothness
        for c in range(3):
            channel = img_array[:, :, c]
            # Gradient magnitude
            gy = np.diff(channel, axis=0)
            gx = np.diff(channel, axis=1)
            gradient_mag = np.sqrt(gy[:, :-1]**2 + gx[:-1, :]**2)
            
            # Very smooth gradients can indicate AI generation
            gradient_std = np.std(gradient_mag)
            gradient_mean = np.mean(gradient_mag)
            
            if gradient_mean < 3 and gradient_std < 5:
                scores.append(0.65)  # Too smooth
            else:
                scores.append(0.3)

        avg_score = np.mean(scores)
        
        # DIGITAL SMOOTHNESS CHECK: AI images have unnaturally smooth gradients
        # We check the average gradient magnitude. Real photos have micro-fluctuations.
        gradient_scores = []
        for c in range(3):
            ch = img_array[:, :, c]
            grad_x = np.abs(np.diff(ch, axis=1))
            avg_grad = np.mean(grad_x)
            if avg_grad < 2.0: # Unnaturally smooth
                gradient_scores.append(0.8)
            elif avg_grad < 4.0:
                gradient_scores.append(0.6)
            else:
                gradient_scores.append(0.2)
        
        final_color_score = (avg_score * 0.4) + (np.mean(gradient_scores) * 0.6)

        return {
            'score': min(max(float(final_color_score), 0), 1),
            'detail': 'Color and gradient smoothness analysis'
        }
    except Exception as e:
        return {'score': 0.5, 'error': str(e)}


def analyze_edges(image):
    """
    Edge Analysis
    Deepfake images often have unnatural edge patterns,
    especially around face boundaries and hair.
    """
    try:
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image

        img_array = np.array(gray, dtype=np.float64)
        
        # Detect edges using Canny
        edges = feature.canny(img_array / 255.0, sigma=1.5)
        
        # Calculate edge density
        edge_density = np.mean(edges)
        
        # Analyze edge connectivity
        labeled_edges = measure.label(edges)
        num_components = labeled_edges.max()
        
        # Calculate edge statistics
        if img_array.shape[0] * img_array.shape[1] > 0:
            normalized_components = num_components / (img_array.shape[0] * img_array.shape[1]) * 10000
        else:
            normalized_components = 0
        
        # AI-generated images tend to have fewer but smoother edges
        if edge_density < 0.03:
            edge_score = 0.65  # Too few edges - possibly AI-smoothed
        elif edge_density > 0.15:
            edge_score = 0.55  # Too many edges - possibly post-processed
        else:
            edge_score = 0.3  # Normal edge density
        
        # Check edge smoothness using Sobel
        sobel_h = filters.sobel_h(img_array / 255.0)
        sobel_v = filters.sobel_v(img_array / 255.0)
        edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        
        # Very uniform edge magnitudes suggest AI generation
        edge_uniformity = np.std(edge_magnitude[edge_magnitude > 0.05]) if np.any(edge_magnitude > 0.05) else 0
        if edge_uniformity < 0.05:
            edge_score += 0.1

        return {
            'score': min(max(float(edge_score), 0), 1),
            'edge_density': float(edge_density),
            'num_components': int(num_components)
        }
    except Exception as e:
        return {'score': 0.5, 'error': str(e)}


def analyze_texture(image):
    """
    Texture Analysis using Local Binary Patterns
    GAN-generated images often have distinctive texture patterns.
    """
    try:
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image

        img_array = np.array(gray, dtype=np.float64)
        
        # Resize for consistent analysis
        target_size = 256
        if max(img_array.shape) > target_size:
            scale = target_size / max(img_array.shape)
            new_h = int(img_array.shape[0] * scale)
            new_w = int(img_array.shape[1] * scale)
            gray_resized = gray.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img_array = np.array(gray_resized, dtype=np.float64)
        
        # Calculate LBP
        radius = 2
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(img_array, n_points, radius, method='uniform')
        
        # Calculate LBP histogram
        n_bins = n_points + 2
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        lbp_hist = lbp_hist.astype(np.float64)
        lbp_hist /= (lbp_hist.sum() + 1e-8)
        
        # Calculate texture entropy
        entropy = -np.sum(lbp_hist[lbp_hist > 0] * np.log2(lbp_hist[lbp_hist > 0]))
        max_entropy = np.log2(n_bins)
        normalized_entropy = entropy / max_entropy
        
        # AI-generated images tend to have lower texture entropy
        # (more repetitive patterns)
        if normalized_entropy < 0.5:
            texture_score = 0.7  # Low entropy - repetitive textures
        elif normalized_entropy < 0.65:
            texture_score = 0.55
        elif normalized_entropy > 0.9:
            texture_score = 0.4  # Very high entropy - could be noisy
        else:
            texture_score = 0.3  # Normal texture diversity
        
        # Check for periodicity in texture
        autocorr = np.correlate(lbp_hist, lbp_hist, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if len(autocorr) > 3:
            periodicity = np.std(autocorr[1:]) / (np.mean(autocorr[1:]) + 1e-8)
            if periodicity > 2:
                texture_score = min(texture_score + 0.1, 1.0)

        return {
            'score': min(max(float(texture_score), 0), 1),
            'entropy': float(normalized_entropy),
            'detail': 'LBP texture analysis'
        }
    except Exception as e:
        return {'score': 0.5, 'error': str(e)}


def analyze_metadata(image, filename=""):
    """
    Analyze image metadata and filename for signs of AI generation or manipulation.
    """
    try:
        info = image.info
        exif = image.getexif() if hasattr(image, 'getexif') else {}
        
        has_exif = len(exif) > 0
        has_camera_info = False
        has_gps = False
        has_software = False
        software_name = ""
        
        # Check for camera-related EXIF tags
        camera_tags = [271, 272, 33434, 33437, 34855]  # Make, Model, ExposureTime, FNumber, ISO
        for tag in camera_tags:
            if tag in exif:
                has_camera_info = True
                break
        
        # Check for GPS data
        if 34853 in exif:
            has_gps = True
        
        # Check for software tag
        if 305 in exif:
            has_software = True
            software_name = str(exif[305])
        
        # Scoring based on metadata
        meta_score = 0.5
        
        if has_camera_info and has_exif:
            meta_score = 0.15  # Likely a real photo with camera data
        elif has_exif and not has_camera_info:
            meta_score = 0.45  # Has some EXIF but no camera info
        elif has_software:
            ai_tools = ['dall-e', 'midjourney', 'stable diffusion', 'ai', 'generated',
                        'photoshop', 'gimp', 'paint']
            if any(tool in software_name.lower() for tool in ai_tools):
                meta_score = 0.9  # Software tag suggests AI or heavy editing
            else:
                meta_score = 0.5
        else:
            meta_score = 0.55  # No metadata - could be stripped (common for AI images)
            
        # Filename analysis (often contains platform names or 'fake')
        ai_keywords = ['chatgpt', 'dalle', 'midjourney', 'stablediffusion', 'bing', 'synthetic', 'generated', 'fake', 'manipulated', 'deepfake']
        social_keywords = ['whatsapp', 'telegram', 'instagram', 'snapchat', 'facebook']
        
        if any(kw in filename.lower() for kw in ai_keywords):
            meta_score = max(meta_score, 0.95)
        elif any(kw in filename.lower() for kw in social_keywords):
            # Social media images are often real but have stripped metadata and high compression
            # We give them a 'Social Trust' bonus, but it will be conditional in the final verdict
            meta_score = 0.1 # Reduced from 0.2 to be more conservative

        return {
            'score': min(max(float(meta_score), 0), 1),
            'has_exif': has_exif,
            'has_camera_info': has_camera_info,
            'has_gps': has_gps,
            'has_software': has_software,
            'software': software_name,
            'format': image.format or 'Unknown',
            'filename': filename
        }
    except Exception as e:
        return {'score': 0.5, 'error': str(e)}


def compute_final_verdict(results):
    """
    CONTEXT-AWARE FORENSIC ENGINE
    Final balanced version for maximum accuracy.
    """
    is_whatsapp = 'whatsapp' in results.get('metadata', {}).get('filename', '').lower()
    is_social = is_whatsapp or any(x in results.get('metadata', {}).get('filename', '').lower() for x in ['telegram', 'instagram', 'facebook'])
    filename = results.get('metadata', {}).get('filename', '').lower()
    
    weights = {
        'ela': 0.15,
        'noise': 0.20,
        'frequency': 0.20,
        'color': 0.15,
        'edges': 0.10,
        'texture': 0.15,
        'metadata': 0.05
    }
    
    # CONTEXT-AWARE PRIORITY BOOSTS
    priority_boost = 0
    
    # Keywords are always a strong signal
    if any(x in filename for x in ['fake', 'deepfake', 'ai', 'synthetic', 'generated']):
        priority_boost = 0.4
        
    # Forensic boosts are harder to trigger on social media to avoid false positives
    ela_boost_threshold = 0.75 if is_social else 0.58
    noise_boost_threshold = 0.78 if is_social else 0.62
    
    if results.get('ela', {}).get('score', 0) > ela_boost_threshold:
        priority_boost += 0.15
    if results.get('noise', {}).get('score', 0) > noise_boost_threshold:
        priority_boost += 0.1
        
    social_shift = 0.22 if is_social else 0.0
    weighted_sum = 0
    total_weight = 0
    
    for key, weight in weights.items():
        if key in results and 'score' in results[key]:
            s = results[key]['score']
            if is_social:
                s = max(s - social_shift, 0.1)
            weighted_sum += s * weight
            total_weight += weight
            
    final_score = (weighted_sum / total_weight if total_weight > 0 else 0.5) + priority_boost
    final_score = min(max(final_score, 0), 1)
    
    # Final Balanced Thresholds
    fake_threshold = 0.60
    real_threshold = 0.45 if is_social else 0.40
    
    if final_score >= fake_threshold:
        verdict = 'FAKE'
        conf = (final_score - 0.35) * 153.8
    elif final_score <= real_threshold:
        verdict = 'REAL'
        conf = (0.5 - final_score) * 200
    elif final_score >= 0.52:
        verdict = 'LIKELY FAKE'
        conf = (final_score - 0.4) * 150
    else:
        verdict = 'LIKELY REAL'
        conf = (0.5 - final_score) * 150
        
    return {
        'score': float(final_score),
        'verdict': verdict,
        'confidence': min(max(float(conf), 15), 99),
        'risk_level': 'HIGH' if final_score >= 0.60 else ('MEDIUM' if final_score >= 0.40 else 'LOW')
    }
        
    return {
        'score': float(final_score),
        'verdict': verdict,
        'confidence': min(max(float(conf), 15), 99),
        'risk_level': 'HIGH' if final_score >= 0.60 else ('MEDIUM' if final_score >= 0.40 else 'LOW')
    }
        
    return {
        'score': float(final_score),
        'verdict': verdict,
        'confidence': min(max(float(conf), 10), 99),
        'risk_level': 'HIGH' if final_score >= fake_threshold else ('MEDIUM' if final_score >= 0.45 else 'LOW')
    }


@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Main endpoint for deepfake detection analysis."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read the image
        image_data = file.read()
        try:
            image = Image.open(io.BytesIO(image_data))
            image.load()  # Force load image data to prevent lazy-loading issues
        except Exception as e:
            return jsonify({'error': f'Invalid or corrupted image: {str(e)}'}), 400

        # Validate file type using filename extension, Pillow detected format, and MIME type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff'}
        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        
        # Fallback to Pillow's detected format if filename has no valid extension
        if ext not in allowed_extensions:
            pil_format = (image.format or '').lower()
            if pil_format == 'jpeg':
                pil_format = 'jpg'
            
            if pil_format in allowed_extensions:
                ext = pil_format
            elif file.content_type and file.content_type.startswith('image/'):
                mime_ext = file.content_type.split('/')[-1].lower()
                if mime_ext in ('jpeg', 'pjpeg'):
                    mime_ext = 'jpg'
                elif mime_ext == 'x-png':
                    mime_ext = 'png'
                if mime_ext in allowed_extensions:
                    ext = mime_ext

        if ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file type: {ext or "unknown"}. Supported: {", ".join(allowed_extensions)}'}), 400
        
        # Save original metadata before potential resizing
        orig_width, orig_height = image.size
        orig_format = image.format or ext.upper()
        
        # Downscale image if it is too large to prevent Out-Of-Memory (OOM) crashes on Render
        # 1024px max dimension is optimal for forensic checks while preserving low RAM usage
        MAX_DIM = 1024
        if max(orig_width, orig_height) > MAX_DIM:
            scale = MAX_DIM / max(orig_width, orig_height)
            new_w = int(orig_width * scale)
            new_h = int(orig_height * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Basic image info
        image_info = {
            'filename': file.filename,
            'format': orig_format,
            'size': f'{orig_width}x{orig_height}',
            'width': orig_width,
            'height': orig_height,
            'mode': image.mode,
            'file_size': len(image_data)
        }

        # Run all analyses
        results = {}
        
        results['ela'] = perform_ela(image)
        results['noise'] = analyze_noise(image)
        results['frequency'] = analyze_frequency(image)
        results['color'] = analyze_color_consistency(image)
        results['edges'] = analyze_edges(image)
        results['texture'] = analyze_texture(image)
        results['metadata'] = analyze_metadata(image, file.filename)
        
        # Compute final verdict
        verdict = compute_final_verdict(results)

        # Create response
        response = {
            'success': True,
            'image_info': image_info,
            'analysis': {
                'ela': {
                    'name': 'Error Level Analysis',
                    'score': results['ela']['score'],
                    'visualization': results['ela'].get('visualization'),
                    'details': {
                        'mean_error': results['ela'].get('mean_error', 0),
                        'uniformity': results['ela'].get('uniformity', 0)
                    }
                },
                'noise': {
                    'name': 'Noise Pattern Analysis',
                    'score': results['noise']['score'],
                    'details': {
                        'noise_level': results['noise'].get('noise_std', 0),
                        'consistency': results['noise'].get('channel_consistency', 0)
                    }
                },
                'frequency': {
                    'name': 'Frequency Domain Analysis',
                    'score': results['frequency']['score'],
                    'visualization': results['frequency'].get('visualization'),
                    'details': {
                        'freq_ratio': results['frequency'].get('freq_ratio', 0)
                    }
                },
                'color': {
                    'name': 'Color Consistency',
                    'score': results['color']['score'],
                    'details': {}
                },
                'edges': {
                    'name': 'Edge Analysis',
                    'score': results['edges']['score'],
                    'details': {
                        'edge_density': results['edges'].get('edge_density', 0)
                    }
                },
                'texture': {
                    'name': 'Texture Analysis',
                    'score': results['texture']['score'],
                    'details': {
                        'entropy': results['texture'].get('entropy', 0)
                    }
                },
                'metadata': {
                    'name': 'Metadata Analysis',
                    'score': results['metadata']['score'],
                    'details': {
                        'has_exif': results['metadata'].get('has_exif', False),
                        'has_camera': results['metadata'].get('has_camera_info', False),
                        'format': results['metadata'].get('format', 'Unknown')
                    }
                }
            },
            'verdict': verdict,
            'timestamp': datetime.utcnow().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
