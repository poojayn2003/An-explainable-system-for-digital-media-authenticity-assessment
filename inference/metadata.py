from pathlib import Path
from PIL import Image, ExifTags


def _format_camera(make, model):
    parts = []

    if make:
        parts.append(str(make).strip())

    if model:
        model = str(model).strip()

        # Avoid duplicate manufacturer name
        if not make or model.lower() not in str(make).lower():
            parts.append(model)

    return " ".join(parts) if parts else "Not Available"


def _format_orientation(value):
    orientation_map = {
        1: "Normal",
        2: "Mirrored horizontally",
        3: "Rotated 180°",
        4: "Mirrored vertically",
        5: "Mirrored + rotated 270Â°",
        6: "Rotated 90Â°",
        7: "Mirrored + rotated 90Â°",
        8: "Rotated 270Â°",
    }

    return orientation_map.get(value, str(value))


def _format_color_space(value):
    if value == 1:
        return "sRGB"

    if value == 2:
        return "Adobe RGB"

    if value is not None:
        return str(value)

    return "Unknown"


def _get_bit_depth(image):
    """
    Return a human-readable bit-depth description.
    This is based on the image mode and format information.
    """

    bits = image.info.get("bits")

    if bits:
        try:
            bits = int(bits)
            channels = len(image.getbands())

            if channels > 1:
                return f"{bits}-bit/channel"
            return f"{bits}-bit"
        except (TypeError, ValueError):
            pass

    mode_depth = {
        "1": "1-bit",
        "L": "8-bit",
        "P": "8-bit",
        "RGB": "8-bit/channel",
        "RGBA": "8-bit/channel",
        "CMYK": "8-bit/channel",
        "I": "32-bit",
        "F": "32-bit",
        "I;16": "16-bit",
        "I;16L": "16-bit",
        "I;16B": "16-bit",
    }

    return mode_depth.get(image.mode, "Unknown")


def _get_compression(image):
    image_format = (image.format or "").upper()

    if image_format == "JPEG":
        return "JPEG (lossy)"

    if image_format == "PNG":
        return "PNG (lossless)"

    if image_format == "WEBP":
        # PIL exposes whether WEBP is lossless through this info field
        if image.info.get("lossless") is True:
            return "WEBP (lossless)"

        return "WEBP"

    if image_format == "TIFF":
        compression = image.info.get("compression")

        if compression:
            return f"TIFF ({compression})"

        return "TIFF"

    return image_format or "Unknown"


def extract_metadata(image_path, original_filename=None):
    """
    Extract forensic metadata from an image.

    Metadata is informational evidence only.
    It does NOT determine whether an image is real or AI-generated.
    """

    path = Path(image_path)

    metadata = {
        "filename": original_filename or path.name,
        "file_type": "Unknown",
        "file_size": round(path.stat().st_size / (1024 * 1024), 2),
        "width": None,
        "height": None,
        "camera": "Not Available",
        "software": "Not Available",
        "date_taken": "Not Available",
        "gps": "Not Available",
        "color_space": "Unknown",
        "orientation": "Unknown",
        "compression": "Unknown",
        "bit_depth": "Unknown",
        "exif_present": False,
        "metadata_status": "No EXIF metadata found",
    }

    try:
        with Image.open(image_path) as image:

            metadata["file_type"] = image.format or "Unknown"
            metadata["width"] = image.width
            metadata["height"] = image.height

            metadata["compression"] = _get_compression(image)
            metadata["bit_depth"] = _get_bit_depth(image)

            exif = image.getexif()

            if not exif:
                return metadata

            metadata["exif_present"] = True
            metadata["metadata_status"] = "EXIF metadata detected"

            exif_data = {}

            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[tag_name] = value

            # Camera
            metadata["camera"] = _format_camera(
                exif_data.get("Make"),
                exif_data.get("Model")
            )

            # Editing/software information
            software = exif_data.get("Software")

            if software:
                metadata["software"] = str(software).strip()

            # Capture date
            date_taken = (
                exif_data.get("DateTimeOriginal")
                or exif_data.get("DateTimeDigitized")
                or exif_data.get("DateTime")
            )

            if date_taken:
                metadata["date_taken"] = str(date_taken)

            # GPS
            if exif_data.get("GPSInfo"):
                metadata["gps"] = "Available"

            # Color space
            metadata["color_space"] = _format_color_space(
                exif_data.get("ColorSpace")
            )

            # Orientation
            orientation = exif_data.get("Orientation")

            if orientation is not None:
                metadata["orientation"] = _format_orientation(orientation)

    except Exception as e:
        metadata["metadata_status"] = "Metadata extraction failed"
        metadata["error"] = str(e)

    return metadata
