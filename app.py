"""Low/High Pass Filter Visualization Application.

This application allows users to apply low/high pass filters to images
and visualize the results using a Gradio-based GUI.
Supports RGB and Grayscale output modes.
"""

import os

import cv2
import gradio as gr
import numpy as np


def create_frequency_mask(
    shape: tuple[int, int], radius: int, filter_type: str
) -> np.ndarray:
    """Create a circular mask for frequency filtering.

    Args:
        shape: Shape of the image (height, width).
        radius: Radius of the filter in pixels.
        filter_type: Either "Low Pass" or "High Pass".

    Returns:
        Binary mask array.
    """
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2

    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)

    if filter_type == "Low Pass":
        mask = distance <= radius
    else:  # High Pass
        mask = distance > radius

    return mask.astype(np.float32)


def apply_frequency_filter_single_channel(
    channel: np.ndarray, radius: int, filter_type: str
) -> np.ndarray:
    """Apply frequency domain filter to a single channel.

    Args:
        channel: Single channel image (grayscale).
        radius: Radius of the filter.
        filter_type: Either "Low Pass" or "High Pass".

    Returns:
        Filtered channel.
    """
    # Ensure radius is an integer
    radius = int(radius)

    # Special case: radius=0 (or near-zero due to slider precision)
    if radius < 1:
        if filter_type == "High Pass":
            # High Pass with radius=0: pass all frequencies → return original
            return channel.astype(np.uint8)
        else:
            # Low Pass with radius=0: only DC component → return mean color
            mean_val = np.mean(channel)
            return np.full_like(channel, mean_val, dtype=np.uint8)

    # Apply FFT
    f_transform = np.fft.fft2(channel)
    f_shift = np.fft.fftshift(f_transform)

    # Create and apply mask
    mask = create_frequency_mask(channel.shape, radius, filter_type)
    f_filtered = f_shift * mask

    # Inverse FFT
    f_ishift = np.fft.ifftshift(f_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)

    # Normalize to 0-255
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)

    return img_filtered


def apply_frequency_filter(
    image: np.ndarray, radius: int, filter_type: str, output_mode: str
) -> np.ndarray:
    """Apply frequency domain filter to an image.

    Args:
        image: Input image (RGB format).
        radius: Radius of the filter.
        filter_type: Either "Low Pass" or "High Pass".
        output_mode: Either "RGB" or "Grayscale".

    Returns:
        Filtered image.
    """
    if output_mode == "Grayscale":
        # Convert to grayscale, apply filter, return grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        return apply_frequency_filter_single_channel(gray, radius, filter_type)
    else:
        # RGB mode: apply filter to each channel separately
        if len(image.shape) == 2:
            # If input is grayscale, convert to RGB first
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Process each channel separately
        channels = []
        for i in range(3):
            filtered_channel = apply_frequency_filter_single_channel(
                image[:, :, i], radius, filter_type
            )
            channels.append(filtered_channel)

        # Merge channels back
        return np.stack(channels, axis=2)


def process_uploaded_image(
    image: np.ndarray | None, radius: int, filter_type: str, output_mode: str
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Process an uploaded image with the specified filter.

    Args:
        image: Input image array (RGB format from Gradio).
        radius: Filter radius.
        filter_type: Either "Low Pass" or "High Pass".
        output_mode: Either "RGB" or "Grayscale".

    Returns:
        Tuple of (original image, filtered image, image info string).
    """
    if image is None:
        return None, None, "Please upload an image"

    # Apply filter
    filtered = apply_frequency_filter(image, radius, filter_type, output_mode)

    # Create image info
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    info = f"""**Image Information**
- Size: {width} x {height} px
- Input Channels: {channels}
- Filter: {filter_type} (Radius: {int(radius)} px)
- Output Mode: {output_mode}"""

    return image, filtered, info


def process_multiple_uploaded_images(
    files: list | None, radius: int, filter_type: str, output_mode: str
) -> tuple[list[tuple], str]:
    """Process multiple uploaded images with the specified filter.

    Args:
        files: List of uploaded file paths.
        radius: Filter radius.
        filter_type: Either "Low Pass" or "High Pass".
        output_mode: Either "RGB" or "Grayscale".

    Returns:
        Tuple of (gallery items, info string).
    """
    if not files:
        return [], "Please upload images"

    gallery_items = []
    info_parts = []

    for file_path in files:
        # Load image from file path
        image = cv2.imread(file_path)
        if image is None:
            continue

        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply filter
        filtered = apply_frequency_filter(image_rgb, radius, filter_type, output_mode)

        # Get filename
        filename = os.path.basename(file_path)

        # Add to gallery (original and filtered)
        gallery_items.append((image_rgb, f"{filename} (Original)"))
        gallery_items.append((filtered, f"{filename} (Filtered)"))

        # Create image info
        height, width = image.shape[:2]
        info_parts.append(f"- **{filename}**: {width}x{height} px")

    info = f"""**Processed {len(files)} images**
Filter: {filter_type} (Radius: {int(radius)} px) | Output: {output_mode}

{chr(10).join(info_parts)}"""

    return gallery_items, info


def create_ui() -> gr.Blocks:
    """Create the Gradio user interface.

    Returns:
        Gradio Blocks application.
    """
    with gr.Blocks(
        title="Low/High Pass Filter Visualization",
    ) as app:
        gr.Markdown("# 🔬 Low/High Pass Filter Visualization")
        gr.Markdown("Visualize frequency filtering using Fourier Transform")

        with gr.Tabs():
            # Single Image Tab
            with gr.TabItem("Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Settings")

                        image_input = gr.Image(
                            label="Upload Image",
                            type="numpy",
                            sources=["upload", "clipboard"],
                            height=200,
                        )

                        filter_type_single = gr.Radio(
                            label="Filter Type",
                            choices=["Low Pass", "High Pass"],
                            value="Low Pass",
                        )

                        output_mode_single = gr.Radio(
                            label="Output Mode",
                            choices=["RGB", "Grayscale"],
                            value="RGB",
                        )

                        radius_slider_single = gr.Slider(
                            label="Filter Radius (px)",
                            minimum=0,
                            maximum=200,
                            value=30,
                            step=1,
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Results")

                        with gr.Row():
                            original_image = gr.Image(
                                label="Original Image",
                                type="numpy",
                                height=400,
                            )
                            filtered_image = gr.Image(
                                label="Filtered Image",
                                type="numpy",
                                height=400,
                            )

                        image_info_single = gr.Markdown("Please upload an image")

            # Multiple Images Tab
            with gr.TabItem("Compare Multiple Images"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Settings")

                        file_upload = gr.File(
                            label="Upload Images",
                            file_count="multiple",
                            file_types=["image"],
                        )

                        filter_type_multi = gr.Radio(
                            label="Filter Type",
                            choices=["Low Pass", "High Pass"],
                            value="Low Pass",
                        )

                        output_mode_multi = gr.Radio(
                            label="Output Mode",
                            choices=["RGB", "Grayscale"],
                            value="RGB",
                        )

                        radius_slider_multi = gr.Slider(
                            label="Filter Radius (px)",
                            minimum=0,
                            maximum=200,
                            value=30,
                            step=1,
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Comparison Gallery")

                        image_gallery = gr.Gallery(
                            label="Original vs Filtered",
                            columns=4,
                            height=500,
                            object_fit="contain",
                        )

                        image_info_multi = gr.Markdown("Upload images to compare")

        # Single Image Event handlers (auto-apply on parameter change)
        filter_type_single.change(
            fn=process_uploaded_image,
            inputs=[
                image_input,
                radius_slider_single,
                filter_type_single,
                output_mode_single,
            ],
            outputs=[original_image, filtered_image, image_info_single],
        )

        output_mode_single.change(
            fn=process_uploaded_image,
            inputs=[
                image_input,
                radius_slider_single,
                filter_type_single,
                output_mode_single,
            ],
            outputs=[original_image, filtered_image, image_info_single],
        )

        radius_slider_single.change(
            fn=process_uploaded_image,
            inputs=[
                image_input,
                radius_slider_single,
                filter_type_single,
                output_mode_single,
            ],
            outputs=[original_image, filtered_image, image_info_single],
        )

        # Also trigger on slider release for reliable update at boundary values
        radius_slider_single.release(
            fn=process_uploaded_image,
            inputs=[
                image_input,
                radius_slider_single,
                filter_type_single,
                output_mode_single,
            ],
            outputs=[original_image, filtered_image, image_info_single],
        )

        image_input.change(
            fn=process_uploaded_image,
            inputs=[
                image_input,
                radius_slider_single,
                filter_type_single,
                output_mode_single,
            ],
            outputs=[original_image, filtered_image, image_info_single],
        )

        # Multiple Images Event handlers (auto-apply on parameter change)
        filter_type_multi.change(
            fn=process_multiple_uploaded_images,
            inputs=[file_upload, radius_slider_multi, filter_type_multi, output_mode_multi],
            outputs=[image_gallery, image_info_multi],
        )

        output_mode_multi.change(
            fn=process_multiple_uploaded_images,
            inputs=[file_upload, radius_slider_multi, filter_type_multi, output_mode_multi],
            outputs=[image_gallery, image_info_multi],
        )

        radius_slider_multi.change(
            fn=process_multiple_uploaded_images,
            inputs=[file_upload, radius_slider_multi, filter_type_multi, output_mode_multi],
            outputs=[image_gallery, image_info_multi],
        )

        # Also trigger on slider release for reliable update at boundary values
        radius_slider_multi.release(
            fn=process_multiple_uploaded_images,
            inputs=[file_upload, radius_slider_multi, filter_type_multi, output_mode_multi],
            outputs=[image_gallery, image_info_multi],
        )

        file_upload.change(
            fn=process_multiple_uploaded_images,
            inputs=[file_upload, radius_slider_multi, filter_type_multi, output_mode_multi],
            outputs=[image_gallery, image_info_multi],
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch()
