import os
import numpy as np
import json
from torchvision.datasets import VisionDataset

from multiview_detector.utils.cameras import build_cam
from multiview_detector.utils.util import get_origin, read_config


class CarlaX(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        # image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
        # CarlaX has xy-indexing: H*W=800*800, thus x is \in [0,800), y \in [0,800)
        # CarlaX has consistent unit: meter (m) for calibration & pos annotation
        self.__name__ = 'CarlaX'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [800, 800]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 4, 400
        self.origin_x, self.origin_y = get_origin(read_config(os.path.join(root, "config.cfg")))
        # world x,y correspond to w,h
        self.indexing = 'xy'
        self.world_indexing_from_xy_mat = np.eye(3)
        self.world_indexing_from_ij_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        # image is in xy indexing by default
        self.img_xy_from_xy_mat = np.eye(3)
        self.img_xy_from_ij_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        # unit in meters
        self.worldcoord_unit = 1
        self.worldcoord_from_worldgrid_mat = np.array([[0.025, 0, self.origin_x], [0, 0.025, self.origin_y], [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 800
        grid_y = pos // 800
        return np.array([[grid_x], [grid_y]], dtype=int).reshape([2, -1])

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid[0, :], worldgrid[1, :]
        return grid_x + grid_y * 800

    def get_worldgrid_from_worldcoord(self, world_coord):
        # datasets default unit: meter & origin determined from cfg
        coord_x, coord_y = world_coord[0, :], world_coord[1, :]
        grid_x = (coord_x - self.origin_x) * 40
        grid_y = (coord_y - self.origin_y) * 40
        return np.array([[grid_x], [grid_y]], dtype=int).reshape([2, -1])

    def get_worldcoord_from_worldgrid(self, worldgrid):
        # datasets default unit: meter & origin determined from cfg
        grid_x, grid_y = worldgrid[0, :], worldgrid[1, :]
        coord_x = self.origin_x + grid_x / 40
        coord_y = self.origin_y + grid_y / 40
        return np.array([[coord_x], [coord_y]]).reshape([2, -1])

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        camera_path = os.path.join(self.root, 'calibrations', f'Camera{camera_i + 1}.json')
        with open(camera_path, "r") as fp:
            camera_values = json.load(fp)
        _, intrinsic_matrix, extrinsic_matrix = build_cam(camera_values)
        return intrinsic_matrix, extrinsic_matrix


def test():
    from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
    dataset = CarlaX(os.path.expanduser('~/Data/CarlaX/01'), )

    # use the first frame for testing
    anno_path = os.path.join(dataset.root, "annotations_positions")
    anno_files = os.path.join(anno_path, sorted(os.listdir(anno_path))[0])
    with open(anno_files, "r") as fp:
        anno = json.load(fp)

    for cam in range(dataset.num_cam):
        head_errors, foot_errors = [], []
        # no pom generated for CarlaX, use pedestrian bbox to perform testing
        for pedestrian in anno:
            cam_view = pedestrian['views'][cam]
            bbox = [cam_view["xmin"], cam_view["ymin"], cam_view["xmax"], cam_view["ymax"]]
            pos = pedestrian['positionID']
            foot_wc = dataset.get_worldcoord_from_pos(pos)
            if all(element == -1 for element in bbox):
                # not visible, skip
                continue
            foot_ic = np.array([[(bbox[0] + bbox[2]) / 2, bbox[3]]]).T
            head_ic = np.array([[(bbox[0] + bbox[2]) / 2, bbox[1]]]).T
            p_foot_wc = get_worldcoord_from_imagecoord(foot_ic, dataset.intrinsic_matrices[cam],
                                                        dataset.extrinsic_matrices[cam])
            p_head_wc = get_worldcoord_from_imagecoord(head_ic, dataset.intrinsic_matrices[cam],
                                                        dataset.extrinsic_matrices[cam], z=1.8 / dataset.worldcoord_unit)
            head_errors.append(np.linalg.norm(p_head_wc - foot_wc))
            foot_errors.append(np.linalg.norm(p_foot_wc - foot_wc))

        print(f'average head error: {np.average(head_errors) * dataset.worldcoord_unit}, '
            f'average foot error: {np.average(foot_errors) * dataset.worldcoord_unit} (world meters)')
        pass
    pass


if __name__ == '__main__':
    test()
