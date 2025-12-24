import cv2
import os
from typing import List, Union

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


def images_to_video(
    image_paths: List[str],
    output_path: str,
    duration_per_image: Union[float, List[float]] = 2.0,
    fps: int = 30,
    target_size: tuple = None,
    use_imageio: bool = None
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
        use_imageio: If True, use imageio; If False, use OpenCV; If None, auto-select (prefer imageio if available)
    """
    if use_imageio is None:
        use_imageio = IMAGEIO_AVAILABLE
    
    if use_imageio:
        if not IMAGEIO_AVAILABLE:
            print("Warning: imageio not available, falling back to OpenCV")
            use_imageio = False
        elif not IMAGEIO_FFMPEG_AVAILABLE:
            print("Warning: imageio-ffmpeg plugin not available, falling back to OpenCV")
            print("To use imageio, install: pip install imageio[ffmpeg] or pip install imageio-ffmpeg")
            use_imageio = False
        else:
            try:
                return images_to_video_with_imageio(
                    image_paths, output_path, duration_per_image, fps, target_size
                )
            except Exception as e:
                print(f"Error using imageio: {e}")
                print("Falling back to OpenCV...")
                use_imageio = False
    
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
    
    for codec_name in codec_priority:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                used_codec = codec_name
                print(f"Using codec: {codec_name}")
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
            
            print(f"Processed: {img_path} (duration: {duration}s)")
        
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
        
        print(f"Processed: {img_path} (duration: {duration}s)")
    
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
    use_imageio: bool = False
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
    """
    image_files = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(image_extensions):
            image_files.append(os.path.join(folder_path, filename))
    
    if not image_files:
        raise ValueError(f"No images found in folder: {folder_path}")
    
    print(f"Found {len(image_files)} images in folder")
    images_to_video(image_files, output_path, duration_per_image, fps, target_size, use_imageio)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert images to MP4 video')
    parser.add_argument('--input', '-i', type=str, help='Input folder or comma-separated image paths')
    parser.add_argument('--output', '-o', type=str, default='output.mp4', help='Output video path')
    parser.add_argument('--duration', '-d', type=float, default=2.0, help='Duration per image in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--width', type=int, help='Video width')
    parser.add_argument('--height', type=int, help='Video height')
    parser.add_argument('--use-imageio', action='store_true', 
                       help='Force use imageio (better Windows compatibility, requires imageio package)')
    parser.add_argument('--use-opencv', action='store_true',
                       help='Force use OpenCV instead of imageio')
    
    args = parser.parse_args()
    
    target_size = None
    if args.width and args.height:
        target_size = (args.width, args.height)
    
    use_imageio = None
    if args.use_imageio:
        use_imageio = True
    elif args.use_opencv:
        use_imageio = False
    
    if os.path.isdir(args.input):
        images_to_video_from_folder(
            args.input,
            args.output,
            args.duration,
            args.fps,
            target_size,
            use_imageio=use_imageio
        )
    else:
        image_paths = [path.strip() for path in args.input.split(',')]
        images_to_video(
            image_paths,
            args.output,
            args.duration,
            args.fps,
            target_size,
            use_imageio
        )

