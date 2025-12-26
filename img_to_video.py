import cv2
import os
import time
import random
import numpy as np
from typing import List, Union, Optional

MOVIEPY_AVAILABLE = False
MOVIEPY_IMPORT_ERROR = None

try:
    import moviepy
    try:
        from moviepy.editor import ImageClip, concatenate_videoclips
        MOVIEPY_AVAILABLE = True
    except ImportError:
        try:
            from moviepy import ImageClip, concatenate_videoclips
            MOVIEPY_AVAILABLE = True
        except ImportError as e:
            MOVIEPY_AVAILABLE = False
            MOVIEPY_IMPORT_ERROR = str(e)
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    MOVIEPY_IMPORT_ERROR = str(e)

try:
    import imageio
    IMAGEIO_AVAILABLE = True
    try:
        import imageio_ffmpeg
        IMAGEIO_FFMPEG_AVAILABLE = True
    except ImportError:
        try:
            imageio.plugins.ffmpeg.get_exe()
            IMAGEIO_FFMPEG_AVAILABLE = True
        except:
            IMAGEIO_FFMPEG_AVAILABLE = False
except ImportError:
    IMAGEIO_AVAILABLE = False
    IMAGEIO_FFMPEG_AVAILABLE = False


def _format_time(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _apply_transition(img1: np.ndarray, img2: np.ndarray, transition_type: str, progress: float, width: int, height: int) -> np.ndarray:
    """
    Apply transition effect between two images.
    
    Args:
        img1: First image (BGR format)
        img2: Second image (BGR format)
        transition_type: Type of transition ('fade', 'slide_left', 'slide_right', 'slide_up', 'slide_down', 
                          'zoom_in', 'zoom_out', 'wipe_left', 'wipe_right', 'rotate')
        progress: Transition progress (0.0 to 1.0)
        width: Target width
        height: Target height
    """
    progress = max(0.0, min(1.0, progress))
    
    if transition_type == 'fade':
        alpha = progress
        result = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
    
    elif transition_type == 'slide_left':
        offset = int(width * progress)
        result = np.zeros_like(img1)
        if offset < width:
            result[:, :width - offset] = img1[:, offset:]
            result[:, width - offset:] = img2[:, :offset]
        else:
            result = img2
    
    elif transition_type == 'slide_right':
        offset = int(width * progress)
        result = np.zeros_like(img1)
        if offset < width:
            result[:, offset:] = img1[:, :width - offset]
            result[:, :offset] = img2[:, width - offset:]
        else:
            result = img2
    
    elif transition_type == 'slide_up':
        offset = int(height * progress)
        result = np.zeros_like(img1)
        if offset < height:
            result[:height - offset, :] = img1[offset:, :]
            result[height - offset:, :] = img2[:offset, :]
        else:
            result = img2
    
    elif transition_type == 'slide_down':
        offset = int(height * progress)
        result = np.zeros_like(img1)
        if offset < height:
            result[offset:, :] = img1[:height - offset, :]
            result[:offset, :] = img2[height - offset:, :]
        else:
            result = img2
    
    elif transition_type == 'zoom_in':
        scale = 1.0 + progress * 0.5
        center_x, center_y = width // 2, height // 2
        M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
        zoomed = cv2.warpAffine(img1, M, (width, height))
        alpha = progress
        result = cv2.addWeighted(zoomed, 1 - alpha, img2, alpha, 0)
    
    elif transition_type == 'zoom_out':
        scale = 1.5 - progress * 0.5
        center_x, center_y = width // 2, height // 2
        M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
        zoomed = cv2.warpAffine(img2, M, (width, height))
        alpha = progress
        result = cv2.addWeighted(img1, 1 - alpha, zoomed, alpha, 0)
    
    elif transition_type == 'wipe_left':
        offset = int(width * progress)
        result = img1.copy()
        result[:, :offset] = img2[:, :offset]
    
    elif transition_type == 'wipe_right':
        offset = int(width * progress)
        result = img1.copy()
        result[:, width - offset:] = img2[:, width - offset:]
    
    elif transition_type == 'rotate':
        angle = progress * 360
        center_x, center_y = width // 2, height // 2
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        rotated = cv2.warpAffine(img1, M, (width, height))
        alpha = progress
        result = cv2.addWeighted(rotated, 1 - alpha, img2, alpha, 0)
    
    else:
        alpha = progress
        result = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
    
    return result


def _get_random_transition() -> str:
    """Get a random transition type."""
    transitions = ['fade', 'slide_left', 'slide_right', 'slide_up', 'slide_down', 
                   'zoom_in', 'zoom_out', 'wipe_left', 'wipe_right', 'rotate']
    return random.choice(transitions)


def _print_progress(current: int, total: int, start_time: float, stage: str = "Processing") -> None:
    if total <= 0:
        return
    elapsed = time.time() - start_time
    progress = current / total
    eta = elapsed / progress - elapsed if progress > 0 else 0.0
    percent = progress * 100.0
    bar_length = 30
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(
        f"\r{stage}: [{bar}] {current}/{total} ({percent:.1f}%) | "
        f"Elapsed: {_format_time(elapsed)} | ETA: {_format_time(eta)}",
        end='', flush=True
    )
    if current >= total:
        print()


def images_to_video_with_moviepy(
    image_paths: List[str],
    output_path: str,
    duration_per_image: Union[float, List[float]] = 2.0,
    fps: int = 30,
    target_size: tuple = None,
    transition_duration: float = 0.5,
    enable_transitions: bool = True
):
    """
    Convert multiple images to MP4 video using MoviePy (best compatibility).
    
    Args:
        image_paths: List of image file paths
        output_path: Output video file path
        duration_per_image: Duration for each image in seconds
        fps: Frames per second for the output video
        target_size: Target resolution (width, height)
        transition_duration: Duration of transition between images in seconds
        enable_transitions: Whether to enable random transitions between images
    """
    print("=" * 60)
    print("Using backend: MoviePy (best compatibility)")
    print("=" * 60)
    
    try:
        try:
            from moviepy.editor import ImageClip, concatenate_videoclips
        except ImportError:
            from moviepy import ImageClip, concatenate_videoclips
    except ImportError as e:
        raise ImportError(f"moviepy is required. Install with: pip install moviepy. Error: {e}")
    
    if not image_paths:
        raise ValueError("No image paths provided")
    
    if isinstance(duration_per_image, (int, float)):
        durations = [float(duration_per_image)] * len(image_paths)
    else:
        if len(duration_per_image) != len(image_paths):
            raise ValueError("Number of durations must match number of images")
        durations = [float(d) for d in duration_per_image]
    
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        raise ValueError(f"Could not read image: {image_paths[0]}")
    
    if target_size is None:
        height, width = first_image.shape[:2]
    else:
        width, height = target_size
    
    total_images = len(image_paths)
    processed_images = 0
    start_time = time.time()
    clips = []
    print(f"Preparing {total_images} image clips...")
    
    prev_img = None
    
    for idx, (img_path, duration) in enumerate(zip(image_paths, durations), 1):
        if not os.path.exists(img_path):
            print(f"\nWarning: Image not found: {img_path}, skipping...")
            continue
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"\nWarning: Could not read image: {img_path}, skipping...")
                continue
            
            if target_size is not None:
                img = cv2.resize(img, (width, height))
            elif img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            
            if enable_transitions and prev_img is not None and idx > 1:
                transition_type = _get_random_transition()
                transition_frames = int(transition_duration * fps)
                
                transition_clips = []
                for frame_idx in range(transition_frames):
                    progress = frame_idx / transition_frames
                    transition_frame = _apply_transition(prev_img, img, transition_type, progress, width, height)
                    transition_frame_rgb = cv2.cvtColor(transition_frame, cv2.COLOR_BGR2RGB)
                    frame_duration = transition_duration / transition_frames
                    transition_clip = ImageClip(transition_frame_rgb, duration=frame_duration)
                    transition_clips.append(transition_clip)
                
                if transition_clips:
                    transition_clip = concatenate_videoclips(transition_clips, method="compose")
                    clips.append(transition_clip)
                    actual_duration = max(0, duration - transition_duration)
                else:
                    actual_duration = duration
            else:
                actual_duration = duration
            
            if actual_duration > 0:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                clip = ImageClip(img_rgb, duration=actual_duration)
                clips.append(clip)
            
            prev_img = img
            processed_images += 1
            _print_progress(processed_images, total_images, start_time, "Preparing clips")
        except Exception as e:
            print(f"\nWarning: Could not process image {img_path}: {e}, skipping...")
            continue
    
    if not clips:
        raise ValueError("No valid clips to concatenate")
    
    final_clip = concatenate_videoclips(clips, method="compose")
    try:
        total_duration = final_clip.duration
        encoding_start = time.time()
        print(f"\nEncoding video (duration: {_format_time(total_duration)})...")
        
        import threading
        import sys
        stop_progress = threading.Event()
        animation_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        animation_index = [0]
        
        def show_progress():
            while not stop_progress.is_set():
                elapsed = time.time() - encoding_start
                elapsed_str = _format_time(elapsed)
                
                animation_index[0] = (animation_index[0] + 1) % len(animation_chars)
                spinner = animation_chars[animation_index[0]]
                
                sys.stdout.write(
                    f"\rEncoding video: {spinner} Elapsed: {elapsed_str}..."
                )
                sys.stdout.flush()
                
                if stop_progress.wait(0.1):
                    break
        
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()
        
        try:
            final_clip.write_videofile(
                output_path,
                fps=fps,
                codec='libx264',
                audio=False,
                preset='medium',
                ffmpeg_params=['-pix_fmt', 'yuv420p'],
                logger=None
            )
        finally:
            stop_progress.set()
            if progress_thread.is_alive():
                progress_thread.join(timeout=1.0)
            elapsed = time.time() - encoding_start
            print(f"\rEncoding video: ✓ Completed | Elapsed: {_format_time(elapsed)}")
        
        encoding_time = time.time() - encoding_start
        print(f"Video saved to: {output_path} (Total encoding time: {_format_time(encoding_time)})")
    finally:
        final_clip.close()
        for clip in clips:
            clip.close()


def images_to_video(
    image_paths: List[str],
    output_path: str,
    duration_per_image: Union[float, List[float]] = 2.0,
    fps: int = 30,
    target_size: tuple = None,
    backend: str = None,
    transition_duration: float = 0.5,
    enable_transitions: bool = True
):
    """
    Convert multiple images to a single MP4 video.
    
    Args:
        image_paths: List of image file paths
        output_path: Output video file path (e.g., 'output.mp4')
        duration_per_image: Duration for each image in seconds. 
                           Can be a single float (same for all) or a list of floats (one per image)
        fps: Frames per second for the output video
        target_size: Target resolution (width, height). If None, uses first image size
        backend: Backend to use: 'moviepy', 'imageio', 'opencv', or None for auto-select
                 Priority: moviepy > imageio > opencv
        transition_duration: Duration of transition between images in seconds
        enable_transitions: Whether to enable random transitions between images
    """
    if backend is None:
        print("\nChecking available backends...")
        
        moviepy_available = MOVIEPY_AVAILABLE
        if not moviepy_available:
            try:
                import moviepy
                try:
                    from moviepy.editor import ImageClip, concatenate_videoclips
                    moviepy_available = True
                    print(f"  MoviePy: ✓ Available (detected at runtime via moviepy.editor)")
                except ImportError:
                    try:
                        from moviepy import ImageClip, concatenate_videoclips
                        moviepy_available = True
                        print(f"  MoviePy: ✓ Available (detected at runtime via moviepy)")
                    except ImportError as e:
                        error_msg = f" (Error: {MOVIEPY_IMPORT_ERROR})" if MOVIEPY_IMPORT_ERROR else f" (ImportError: {e})"
                        print(f"  MoviePy: ✗ Not available{error_msg}")
            except ImportError as e:
                error_msg = f" (Error: {MOVIEPY_IMPORT_ERROR})" if MOVIEPY_IMPORT_ERROR else f" (ImportError: {e})"
                print(f"  MoviePy: ✗ Not available{error_msg}")
        else:
            print(f"  MoviePy: ✓ Available")
        
        print(f"  ImageIO: {'✓ Available' if IMAGEIO_AVAILABLE else '✗ Not available'}")
        if IMAGEIO_AVAILABLE:
            print(f"  ImageIO-FFmpeg: {'✓ Available' if IMAGEIO_FFMPEG_AVAILABLE else '✗ Not available'}")
        print(f"  OpenCV: ✓ Available (fallback)\n")
        
        if moviepy_available:
            backend = 'moviepy'
        elif IMAGEIO_AVAILABLE and IMAGEIO_FFMPEG_AVAILABLE:
            backend = 'imageio'
        else:
            backend = 'opencv'
        print(f"Auto-selected backend: {backend.upper()}\n")
    
    if backend == 'moviepy':
        try:
            try:
                from moviepy.editor import ImageClip, concatenate_videoclips
            except ImportError:
                from moviepy import ImageClip, concatenate_videoclips
            return images_to_video_with_moviepy(
                image_paths, output_path, duration_per_image, fps, target_size, transition_duration, enable_transitions
            )
        except ImportError as e:
            print(f"Warning: moviepy not available ({e}), trying other backends...")
            backend = 'imageio' if (IMAGEIO_AVAILABLE and IMAGEIO_FFMPEG_AVAILABLE) else 'opencv'
            print(f"Switching to backend: {backend.upper()}")
        except Exception as e:
            print(f"Error using moviepy: {e}")
            print("Falling back to other backends...")
            backend = 'imageio' if (IMAGEIO_AVAILABLE and IMAGEIO_FFMPEG_AVAILABLE) else 'opencv'
            print(f"Switching to backend: {backend.upper()}")
    
    if backend == 'imageio':
        if not IMAGEIO_AVAILABLE:
            print("Warning: imageio not available, falling back to OpenCV")
            backend = 'opencv'
            print(f"Switching to backend: {backend.upper()}")
        elif not IMAGEIO_FFMPEG_AVAILABLE:
            print("Warning: imageio-ffmpeg plugin not available, falling back to OpenCV")
            print("To use imageio, install: pip install imageio[ffmpeg] or pip install imageio-ffmpeg")
            backend = 'opencv'
            print(f"Switching to backend: {backend.upper()}")
        else:
            try:
                return images_to_video_with_imageio(
                    image_paths, output_path, duration_per_image, fps, target_size, transition_duration, enable_transitions
                )
            except Exception as e:
                print(f"Error using imageio: {e}")
                print("Falling back to OpenCV...")
                backend = 'opencv'
                print(f"Switching to backend: {backend.upper()}")
    
    print("=" * 60)
    print("Using backend: OpenCV")
    print("=" * 60)
    
    if not image_paths:
        raise ValueError("No image paths provided")
    
    if isinstance(duration_per_image, (int, float)):
        durations = [float(duration_per_image)] * len(image_paths)
    else:
        if len(duration_per_image) != len(image_paths):
            raise ValueError("Number of durations must match number of images")
        durations = [float(d) for d in duration_per_image]
    
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        raise ValueError(f"Could not read image: {image_paths[0]}")
    
    if target_size is None:
        height, width = first_image.shape[:2]
    else:
        width, height = target_size
    
    codec_priority = ['XVID', 'MJPG', 'X264', 'avc1', 'H264', 'mp4v']
    out = None
    used_codec = None
    
    print("Trying codecs:", ', '.join(codec_priority))
    for codec_name in codec_priority:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                used_codec = codec_name
                print(f"Selected codec: {codec_name}")
                break
            else:
                out.release()
        except:
            continue
    
    if out is None or not out.isOpened():
        raise RuntimeError(
            f"Failed to initialize video writer. "
            f"Tried codecs: {', '.join(codec_priority)}. "
            f"Please ensure OpenCV is properly installed with codec support."
        )
    
    total_images = len(image_paths)
    processed_images = 0
    start_time = time.time()
    print(f"Encoding {total_images} images to video...")
    
    prev_img = None
    
    try:
        for idx, (img_path, duration) in enumerate(zip(image_paths, durations), 1):
            if not os.path.exists(img_path):
                print(f"\nWarning: Image not found: {img_path}, skipping...")
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"\nWarning: Could not read image: {img_path}, skipping...")
                continue
            
            if target_size is not None:
                img = cv2.resize(img, (width, height))
            elif img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            
            if enable_transitions and prev_img is not None and idx > 1:
                transition_type = _get_random_transition()
                transition_frames = int(transition_duration * fps)
                
                for frame_idx in range(transition_frames):
                    progress = frame_idx / transition_frames
                    transition_frame = _apply_transition(prev_img, img, transition_type, progress, width, height)
                    out.write(transition_frame)
                
                actual_duration = max(0, duration - transition_duration)
            else:
                actual_duration = duration
            
            frames_for_image = int(actual_duration * fps)
            for _ in range(frames_for_image):
                out.write(img)
            
            prev_img = img
            processed_images += 1
            _print_progress(processed_images, total_images, start_time, "Encoding video")
        
        print(f"\nVideo saved to: {output_path}")
    
    finally:
        out.release()


def images_to_video_with_imageio(
    image_paths: List[str],
    output_path: str,
    duration_per_image: Union[float, List[float]] = 2.0,
    fps: int = 30,
    target_size: tuple = None,
    transition_duration: float = 0.5,
    enable_transitions: bool = True
):
    """
    Convert multiple images to MP4 video using imageio (better Windows compatibility).
    
    Args:
        image_paths: List of image file paths
        output_path: Output video file path
        duration_per_image: Duration for each image in seconds
        fps: Frames per second for the output video
        target_size: Target resolution (width, height)
        transition_duration: Duration of transition between images in seconds
        enable_transitions: Whether to enable random transitions between images
    """
    print("=" * 60)
    print("Using backend: ImageIO")
    print("=" * 60)
    
    if not IMAGEIO_AVAILABLE:
        raise ImportError("imageio is required. Install with: pip install imageio imageio-ffmpeg")
    
    if not image_paths:
        raise ValueError("No image paths provided")
    
    if isinstance(duration_per_image, (int, float)):
        durations = [float(duration_per_image)] * len(image_paths)
    else:
        if len(duration_per_image) != len(image_paths):
            raise ValueError("Number of durations must match number of images")
        durations = [float(d) for d in duration_per_image]
    
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        raise ValueError(f"Could not read image: {image_paths[0]}")
    
    if target_size is None:
        height, width = first_image.shape[:2]
    else:
        width, height = target_size
    
    total_images = len(image_paths)
    processed_images = 0
    start_time = time.time()
    frames = []
    print(f"Preparing {total_images} image frames...")
    
    prev_img = None
    
    for idx, (img_path, duration) in enumerate(zip(image_paths, durations), 1):
        if not os.path.exists(img_path):
            print(f"\nWarning: Image not found: {img_path}, skipping...")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"\nWarning: Could not read image: {img_path}, skipping...")
            continue
        
        if target_size is not None:
            img = cv2.resize(img, (width, height))
        elif img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        if enable_transitions and prev_img is not None and idx > 1:
            transition_type = _get_random_transition()
            transition_frames = int(transition_duration * fps)
            
            for frame_idx in range(transition_frames):
                progress = frame_idx / transition_frames
                transition_frame = _apply_transition(prev_img, img, transition_type, progress, width, height)
                transition_frame_rgb = cv2.cvtColor(transition_frame, cv2.COLOR_BGR2RGB)
                frames.append(transition_frame_rgb)
            
            actual_duration = max(0, duration - transition_duration)
        else:
            actual_duration = duration
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames_for_image = int(actual_duration * fps)
        for _ in range(frames_for_image):
            frames.append(img_rgb)
        
        prev_img = img
        processed_images += 1
        _print_progress(processed_images, total_images, start_time, "Preparing frames")
    
    if not frames:
        raise ValueError("No valid frames to write")
    
    try:
        total_frames = len(frames)
        estimated_duration = total_frames / fps
        encoding_start = time.time()
        print(f"\nEncoding video ({total_frames} frames, estimated duration: {_format_time(estimated_duration)})...")
        
        import threading
        import sys
        stop_progress = threading.Event()
        animation_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        animation_index = [0]
        
        def show_progress():
            while not stop_progress.is_set():
                elapsed = time.time() - encoding_start
                elapsed_str = _format_time(elapsed)
                
                animation_index[0] = (animation_index[0] + 1) % len(animation_chars)
                spinner = animation_chars[animation_index[0]]
                
                sys.stdout.write(
                    f"\rEncoding video: {spinner} Elapsed: {elapsed_str}..."
                )
                sys.stdout.flush()
                
                if stop_progress.wait(0.1):
                    break
        
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()
        
        try:
            imageio.mimwrite(
                output_path, 
                frames, 
                fps=fps, 
                codec='libx264', 
                quality=8
            )
        finally:
            stop_progress.set()
            if progress_thread.is_alive():
                progress_thread.join(timeout=1.0)
            elapsed = time.time() - encoding_start
            print(f"\rEncoding video: ✓ Completed | Elapsed: {_format_time(elapsed)}")
        
        encoding_time = time.time() - encoding_start
        print(f"Video saved to: {output_path} (Total encoding time: {_format_time(encoding_time)})")
    except Exception as e:
        error_msg = str(e)
        if 'ffmpeg' in error_msg.lower() or 'backend' in error_msg.lower():
            raise ImportError(
                f"imageio-ffmpeg plugin not available. "
                f"Install with: pip install imageio[ffmpeg] or pip install imageio-ffmpeg"
            )
        raise


def images_to_video_from_folder(
    folder_path: str,
    output_path: str,
    duration_per_image: Union[float, List[float]] = 2.0,
    fps: int = 30,
    target_size: tuple = None,
    image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
    backend: str = None,
    transition_duration: float = 0.5,
    enable_transitions: bool = True
):
    """
    Convert all images in a folder to a single MP4 video.
    
    Args:
        folder_path: Path to folder containing images
        output_path: Output video file path
        duration_per_image: Duration for each image in seconds
        fps: Frames per second for the output video
        target_size: Target resolution (width, height)
        image_extensions: Tuple of valid image file extensions
        backend: Backend to use: 'moviepy', 'imageio', 'opencv', or None for auto-select
        transition_duration: Duration of transition between images in seconds
        enable_transitions: Whether to enable random transitions between images
    """
    image_files = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(image_extensions):
            image_files.append(os.path.join(folder_path, filename))
    
    if not image_files:
        raise ValueError(f"No images found in folder: {folder_path}")
    
    print(f"Found {len(image_files)} images in folder")
    images_to_video(image_files, output_path, duration_per_image, fps, target_size, backend, transition_duration, enable_transitions)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert images to MP4 video')
    parser.add_argument('--input', '-i', type=str, help='Input folder or comma-separated image paths')
    parser.add_argument('--output', '-o', type=str, default='output.mp4', help='Output video path')
    parser.add_argument('--duration', '-d', type=float, default=2.0, help='Duration per image in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--width', type=int, help='Video width')
    parser.add_argument('--height', type=int, help='Video height')
    parser.add_argument('--backend', type=str, choices=['moviepy', 'imageio', 'opencv'],
                       help='Backend to use: moviepy (best compatibility), imageio, or opencv. Auto-select if not specified.')
    parser.add_argument('--use-moviepy', action='store_true',
                       help='Force use MoviePy (best compatibility, requires moviepy package)')
    parser.add_argument('--use-imageio', action='store_true',
                       help='Force use imageio (requires imageio package)')
    parser.add_argument('--use-opencv', action='store_true',
                       help='Force use OpenCV')
    parser.add_argument('--transition-duration', '-t', type=float, default=0.5,
                       help='Duration of transition between images in seconds (default: 0.5)')
    parser.add_argument('--no-transitions', action='store_true',
                       help='Disable transitions between images')
    
    args = parser.parse_args()
    
    target_size = None
    if args.width and args.height:
        target_size = (args.width, args.height)
    
    backend = None
    if args.backend:
        backend = args.backend
    elif args.use_moviepy:
        backend = 'moviepy'
    elif args.use_imageio:
        backend = 'imageio'
    elif args.use_opencv:
        backend = 'opencv'
    
    enable_transitions = not args.no_transitions
    
    if os.path.isdir(args.input):
        images_to_video_from_folder(
            args.input,
            args.output,
            args.duration,
            args.fps,
            target_size,
            backend=backend,
            transition_duration=args.transition_duration,
            enable_transitions=enable_transitions
        )
    else:
        image_paths = [path.strip() for path in args.input.split(',')]
        images_to_video(
            image_paths,
            args.output,
            args.duration,
            args.fps,
            target_size,
            backend,
            args.transition_duration,
            enable_transitions
        )

