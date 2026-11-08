import numpy as np
import cv2
from pathlib import Path

def create_sample_sketch(output_path: Path, width: int = 640, height: int = 480) -> Path:
    """Creates a synthetic set design sketch of a classical theater stage with stairs, columns, and arch."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # White background with subtle paper tone
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    
    # Charcoal / pencil stroke color
    pencil = (35, 30, 25)

    # Stage floor perspective lines
    cv2.line(img, (50, 420), (width - 50, 420), pencil, 3)
    cv2.line(img, (50, 420), (160, 320), pencil, 2)
    cv2.line(img, (width - 50, 420), (width - 160, 320), pencil, 2)
    cv2.line(img, (160, 320), (width - 160, 320), pencil, 2)

    # Central arch
    cv2.ellipse(img, (width // 2, 220), (120, 100), 0, 180, 360, pencil, 3)
    cv2.line(img, (width // 2 - 120, 220), (width // 2 - 120, 320), pencil, 3)
    cv2.line(img, (width // 2 + 120, 220), (width // 2 + 120, 320), pencil, 3)

    # Columns left and right
    for col_x in [120, width - 150]:
        cv2.rectangle(img, (col_x, 140), (col_x + 30, 380), pencil, 2)
        # Column grooves
        cv2.line(img, (col_x + 10, 150), (col_x + 10, 370), pencil, 1)
        cv2.line(img, (col_x + 20, 150), (col_x + 20, 370), pencil, 1)
        # Capital and base
        cv2.rectangle(img, (col_x - 5, 130), (col_x + 35, 140), pencil, 2)
        cv2.rectangle(img, (col_x - 5, 380), (col_x + 35, 395), pencil, 2)

    # Stage stairs
    for y_step in [380, 395, 410]:
        cv2.line(img, (200, y_step), (width - 200, y_step), pencil, 2)

    # Background window / backdrop
    cv2.rectangle(img, (240, 160), (width - 240, 260), pencil, 1)

    cv2.imwrite(str(output_path), img)
    return output_path

if __name__ == "__main__":
    create_sample_sketch(Path("samples/sample_stage_sketch.png"))
    print("Sample sketch generated at samples/sample_stage_sketch.png")
