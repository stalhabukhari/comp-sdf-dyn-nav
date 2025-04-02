import os
from pathlib import Path
from typing import Dict

import numpy as np
import trimesh

from data import common as cmn
from data import mesh_utils


base_dir = Path(__file__).parent.parent
YOLO_LABEL2NAME_DICT = cmn.parse_yaml_file(
    base_dir / "configs/yolo_label_to_obj_map.yaml"
)
YOLO_NAME2IGOBJ_DICT = cmn.parse_yaml_file(
    base_dir / "configs/yolo_obj_to_ig_obj_map.yaml"
)


class MiscDataHandler:
    def __init__(self, misc_config_dict: Dict):
        """
        in misc_config_dict:
        igibson_data_dir: this should contain "ig_dataset" folder
        deepsdf_data_dir: this should contain object category folders (e.g, "armchair")
        """
        self.ig_data_dir = Path(misc_config_dict["igibson_data_dir"])
        self.deepsdf_data_dir = Path(misc_config_dict["deepsdf_data_dir"])
        # for object scale computation
        ig_avg_category_specs_json_filepath = (
            self.ig_data_dir / "ig_dataset" / "metadata" / "avg_category_specs.json"
        )
        self.avg_category_specs = cmn.load_json(ig_avg_category_specs_json_filepath)

    @staticmethod
    def get_object_name_for_label(label: int) -> str:
        return YOLO_LABEL2NAME_DICT[label]

    @staticmethod
    def get_ig_object_for_name(name: str) -> Dict:
        # returns a dict of ("name", "instance")
        return YOLO_NAME2IGOBJ_DICT[name]

    def get_ig_object_for_label(self, label: int) -> Dict:
        # returns a dict of ("name", "instance")
        name = self.get_object_name_for_label(label=label)
        return self.get_ig_object_for_name(name=name)

    def get_mesh_norm_params_for_object(
        self, object_category: str, object_model: str
    ) -> mesh_utils.MeshNormParams:
        json_filepath = (
            self.deepsdf_data_dir
            / object_category
            / object_model
            / f"{object_model}-norm-params.json"
        )
        norm_params = mesh_utils.MeshNormParams.from_json_file(json_filepath)
        return norm_params

    def get_object_bbox_size(
        self, object_category: str, object_model: str
    ) -> np.ndarray:
        meta_json = (
            self.ig_data_dir
            / "ig_dataset"
            / "objects"
            / object_category
            / object_model
            / "misc"
            / "metadata.json"
        )
        metadata = cmn.load_json(meta_json)
        # used in object scale computation
        bbox_size = np.array(metadata["bbox_size"])
        return bbox_size

    @staticmethod
    def get_object_category_instance_for_body_id(env, body_id: int) -> Dict:
        """
        env is an `iGibsonEnv` object
        Can move MiscDataHandler to a different package to avoid circular dependencies.
        """
        # below line copied from lib_igibson
        object_urdf = env.scene.objects_by_id[body_id]

        object_category = object_urdf.category
        _object_model = Path(object_urdf.filename).stem.split("_")[0]

        # hack to get the correct object instance
        object_instance = None
        for objname, objdata in YOLO_NAME2IGOBJ_DICT.items():
            _name = objdata["name"]
            _inst = objdata["instance"]
            if object_category == _name and _object_model in _inst:
                object_instance = _inst
                break

        assert object_instance is not None, (
            f"Could not find appropriate instance for an object "
            f"of category {object_category}"
        )
        data_dict = {"category": object_category, "instance": object_instance}
        return data_dict

    def get_yolo_label_for_body_id(self, env, body_id: int) -> int:
        data_dict = self.get_object_category_instance_for_body_id(
            env=env, body_id=body_id
        )
        object_category = data_dict["category"]
        object_instance = data_dict["instance"]
        object_name = f"{object_category}|{object_instance}"
        for _label, _obj_name in YOLO_LABEL2NAME_DICT.items():
            if _obj_name == object_name:
                return _label
        raise Exception(
            f"Could not find YOLO label for body_id: {body_id}"
        )  # should never happen

    def get_object_bbox_size_scaled(
        self, object_category: str, object_model: str
    ) -> np.ndarray:
        # get object category specs
        avg_obj_dims = self.avg_category_specs.get(object_category, None)
        # read object metadata
        bbox_size = self.get_object_bbox_size(
            object_category=object_category, object_model=object_model
        )
        if avg_obj_dims is None:
            scale = np.ones(3)
        else:
            spec_vol = (
                avg_obj_dims["size"][0]
                * avg_obj_dims["size"][1]
                * avg_obj_dims["size"][2]
            )
            curr_vol = bbox_size[0] * bbox_size[1] * bbox_size[2]
            volume_ratio = spec_vol / curr_vol
            size_ratio = np.cbrt(volume_ratio)
            scale = np.array([size_ratio] * 3)
        return bbox_size * scale

    def compute_igibson_scaling_for_object(
        self, object_category: str, object_model: str
    ) -> np.ndarray:
        # For real world
        # return np.eye(4)

        # get object category specs
        avg_obj_dims = self.avg_category_specs.get(object_category, None)
        # read object metadata
        bbox_size = self.get_object_bbox_size(
            object_category=object_category, object_model=object_model
        )
        # compute scales
        if avg_obj_dims is None:
            scale = np.ones(3)
        else:
            spec_vol = (
                avg_obj_dims["size"][0]
                * avg_obj_dims["size"][1]
                * avg_obj_dims["size"][2]
            )
            curr_vol = bbox_size[0] * bbox_size[1] * bbox_size[2]
            volume_ratio = spec_vol / curr_vol
            size_ratio = np.cbrt(volume_ratio)
            scale = np.array([size_ratio] * 3)

        # scale as a homogeneous transformation
        scale_mat = np.diag(scale)
        scale_mat_H = np.eye(4)
        scale_mat_H[:3, :3] = scale_mat
        return scale_mat_H


def get_object_volume(env, body_id) -> float:
    """
    calculate volume of an object by combining volume of its parts.
    Uses trimesh.
    """
    object_urdf = env.scene.objects_by_id[body_id]
    all_links = object_urdf.object_tree.findall("link")

    total_volume = 0.0
    for link in all_links:
        meshes = link.findall("collision/geometry/mesh")
        if len(meshes) == 0:
            continue
        # assume one collision mesh per link
        assert len(meshes) == 1, (object_urdf.filename, link.attrib["name"])
        # check collision mesh path
        collision_mesh_path = os.path.join(meshes[0].attrib["filename"])
        trimesh_obj = trimesh.load(file_obj=collision_mesh_path, force="mesh")
        volume = trimesh_obj.volume
        total_volume += volume
    return total_volume
