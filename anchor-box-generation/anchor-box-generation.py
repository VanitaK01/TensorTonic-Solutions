import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    stride = image_size / feature_size
    anchors = []

    # Iterate grid (row-major: i then j)
    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            # Iterate scales then aspect ratios
            for s in scales:
                for r in aspect_ratios:
                    w = s * np.sqrt(r)
                    h = s / np.sqrt(r)

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    anchors.append([float(x1), float(y1), float(x2), float(y2)])

    return anchors