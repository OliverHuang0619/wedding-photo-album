import cv2
import os
import time
from typing import List, Union

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


def _print_progress(prefix: str, current: int, total: int, start_time: float) -> None:
    if total <= 0:
        return
    elapsed = time.time() - start_time
    progress = current / total
    eta = elapsed / progress - elapsed if progress > 0 else 0.0
    percent = progress * 100.0
    print(
        f"{prefix}: {current}/{total} ({percent:.1f}%) "
        f"Elapsed: {_format_time(elapsed)} "
        f"ETA: {_format_time(eta)}"
    )


def images_to_video_with_moviepy(
    image_paths: List[str],
    output_path: str,
    duration_per_image: Union[float, List[float]] = 2.0,
    fps: int = 30,
    target_size: tuple = None
):
    """
    Convert multiple images to MP4 video using MoviePy (best compatibility).
    
    Args:
        image_paths: List of image file paths
        output_path: Output video file path
        duration_per_image: Duration for each image in seconds
        fps: Frames per second for the output video
        target_size: Target resolution (width, height)
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
    for img_path, duration in zip(image_paths, durations):
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}, skipping...")
            continue
        
        try:
            clip = ImageClip(img_path, duration=duration)
            if target_size is not None or clip.size != (width, height):
                try:
                    clip = clip.resize(width=width, height=height)
                except (TypeError, AttributeError):
                    try:
                        clip = clip.resized(size=(width, height))
                    except (TypeError, AttributeError):
                        try:
                            clip = clip.resized(newsize=(width, height))
                        except (TypeError, AttributeError):
                            try:
                                clip = clip.with_size((width, height))
                            except (TypeError, AttributeError):
                                clip = clip.resized((width, height))
            clips.append(clip)
            processed_images += 1
            print(f"Processed: {img_path} (duration: {duration}s)")
            _print_progress(
                "Preparing clips",
                processed_images,
                total_images,
                start_time
            )
        except Exception as e:
            print(f"Warning: Could not process image {img_path}: {e}, skipping...")
            continue
    
    if not clips:
        raise ValueError("No valid clips to concatenate")
    
    final_clip = concatenate_videoclips(clips, method="compose")
    try:
        final_clip.write_videofile(
            output_path,
            fps=fps,
            codec='libx264',
            audio=False,
            preset='medium',
            ffmpeg_params=['-pix_fmt', 'yuv420p'],
            logger="bar"
        )
        print(f"Video saved to: {output_path}")
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
    backend: str = None
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
                image_paths, output_path, duration_per_image, fps, target_size
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
                    image_paths, output_path, duration_per_image, fps, target_size
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
    try:
        for img_path, duration in zip(image_paths, durations):
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}, skipping...")
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image: {img_path}, skipping...")
                continue
            
            if target_size is not None:
                img = cv2.resize(img, (width, height))
            elif img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            
            frames_for_image = int(duration * fps)
            for _ in range(frames_for_image):
                out.write(img)
            processed_images += 1
            print(f"Processed: {img_path} (duration: {duration}s)")
            _print_progress(
                "Encoding video (OpenCV)",
                processed_images,
                total_images,
                start_time
            )
        
        print(f"Video saved to: {output_path}")
    
    finally:
        out.release()


def images_to_video_with_imageio(
    image_paths: List[str],
    output_path: str,
    duration_per_image: Union[float, List[float]] = 2.0,
    fps: int = 30,
    target_size: tuple = None
):
    """
    Convert multiple images to MP4 video using imageio (better Windows compatibility).
    
    Args:
        image_paths: List of image file paths
        output_path: Output video file path
        duration_per_image: Duration for each image in seconds
        fps: Frames per second for the output video
        target_size: Target resolution (width, height)
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
    for img_path, duration in zip(image_paths, durations):
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}, skipping...")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image: {img_path}, skipping...")
            continue
        
        if target_size is not None:
            img = cv2.resize(img, (width, height))
        elif img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames_for_image = int(duration * fps)
        for _ in range(frames_for_image):
            frames.append(img_rgb)
        processed_images += 1
        print(f"Processed: {img_path} (duration: {duration}s)")
        _print_progress(
            "Preparing frames (ImageIO)",
            processed_images,
            total_images,
            start_time
        )
    
    if not frames:
        raise ValueError("No valid frames to write")
    
    try:
        imageio.mimwrite(
            output_path, 
            frames, 
            fps=fps, 
            codec='libx264', 
            quality=8
        )
        print(f"Video saved to: {output_path}")
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
    backend: str = None
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
    """
    image_files = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(image_extensions):
            image_files.append(os.path.join(folder_path, filename))
    
    if not image_files:
        raise ValueError(f"No images found in folder: {folder_path}")
    
    print(f"Found {len(image_files)} images in folder")
    images_to_video(image_files, output_path, duration_per_image, fps, target_size, backend)


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
    
    if os.path.isdir(args.input):
        images_to_video_from_folder(
            args.input,
            args.output,
            args.duration,
            args.fps,
            target_size,
            backend=backend
        )
    else:
        image_paths = [path.strip() for path in args.input.split(',')]
        images_to_video(
            image_paths,
            args.output,
            args.duration,
            args.fps,
            target_size,
            backend
        )

