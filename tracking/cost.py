import numpy as np
from shapely.geometry import Polygon


def create_polygon(bboxes: np.ndarray) -> Polygon:
    x, y, l, w, yaw = bboxes
    # tl: (x-l/2, y+w/2), bl: (x-l/2, y-w/2), br: (x+l/2, y-w/2), tr: (x+l/2, y+w/2)
    # Rotate about (p, q): x' = (x-p)*cos(yaw) - (y-q)*sin(yaw) + p, y' = (x-p)*sin(yaw) + (y-q)*cos(yaw) + q
    sin, cos = np.sin(yaw), np.cos(yaw)
    return Polygon([
                (((x-l/2)-x)*cos - ((y+w/2)-y)*sin + x, ((x-l/2)-x)*sin + ((y+w/2)-y)*cos) + y,
                (((x-l/2)-x)*cos - ((y-w/2)-y)*sin + x, ((x-l/2)-x)*sin + ((y-w/2)-y)*cos) + y,
                (((x+l/2)-x)*cos - ((y-w/2)-y)*sin + x, ((x+l/2)-x)*sin + ((y-w/2)-y)*cos) + y,
                (((x+l/2)-x)*cos - ((y+w/2)-y)*sin + x, ((x+l/2)-x)*sin + ((y+w/2)-y)*cos) + y
            ])


def iou_2d(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes 2D intersection over union of two sets of bounding boxes

    Args:
        bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
        bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
    Returns:
        iou_mat: matrix of shape [M, N], where iou_mat[i, j] is the 2D IoU value between bboxes[i] and bboxes[j].
        You should use the Polygon class from the shapely package to compute the area of intersection/union.
    """
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    # DONE: Replace this stub code.
    iou_mat = -np.ones((M, N))
    for i in range(M):
        polygon1 = create_polygon(bboxes1[i])
        for j in range(N):
            if iou_mat[i, j] != -1.0:  # ignore duplicate calculation
                continue
            polygon2 = create_polygon(bboxes2[j])
            area_of_overlap = polygon1.intersection(polygon2).area
            area_of_union = polygon1.union(polygon2).area
            iou = area_of_overlap / area_of_union
            iou_mat[i, j] = iou
            if j < M:
                iou_mat[j, i] = iou  # reduce workload, since (i, j) and (j, i) should have same value
    return iou_mat


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     bboxes_a = np.array([[0.0, 0.0, 2.0, 1.0, 0.0], [0.0, 0.0, 2.0, 1.0, np.pi / 2]])
#     bboxes_b = np.array([[1.0, 0.0, 2.0, 1.0, 0.0], [1.0, 0.0, 2.0, 1.0, np.pi / 2]])
#     polygon1 = create_polygon(bboxes_a[1])
#     polygon2 = create_polygon(bboxes_b[1])
#     plt.plot(*polygon1.exterior.xy)
#     plt.plot(*polygon2.exterior.xy)
#     plt.show()
#     iou_mat = iou_2d(bboxes_a, bboxes_b)
#     print(iou_mat)
