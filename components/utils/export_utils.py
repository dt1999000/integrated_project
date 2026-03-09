from typing import Any, Dict, List


class Export:
    """
    Helper utilities for dataset exports.

    This class centralizes small transformation helpers so that behaviour
    can be changed in one place (for example when external tools like CVAT
    change their conventions).
    """

    @staticmethod
    def swap_cvat_dimensions(size: List[float]) -> List[float]:
        """
        Swap length/width/height before exporting to CVAT/Datumaro JSON.

        Notes
        -----
        CVAT's 3D cuboid UI currently interprets the scale components with a
        non-standard axis convention compared to the rest of this project:

            - Normal convention in this project:
              length -> X, width -> Y, height -> Z

            - CVAT UI observation:
              width  -> X, height -> Y, length -> Z

        To compensate, this helper reorders the components so that the
        resulting cuboids look correct in CVAT. If CVAT fixes its UI bug,
        this function can be changed back to an identity mapping without
        touching all call sites.
        """
        if len(size) != 3:
            return size

        length, width, height = size

        # Current workaround mapping:
        #   internal (L, W, H) -> exported (W, H, L)
        # so that CVAT's (width, height, length) axes end up with the
        # expected physical dimensions.
        return [width, height, length]

    @staticmethod
    def reverse_frame_order(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Return a new list with frame entries in reversed temporal order.

        Some KITTI-style exports currently appear with frames reversed
        (last frame first). Calling this helper just before writing the
        export ensures a consistent chronological ordering.
        """
        return list(reversed(frames))

