from typing import List

import numpy as np


def get_bbox_from_mask(mask: np.ndarray) -> List[int]:
    """
    Compute axis-aligned bbox [x_min, y_min, x_max, y_max] from a binary mask.
    Returns [0, 0, 0, 0] when the mask has no positive pixels.
    """
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return [0, 0, 0, 0]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]
