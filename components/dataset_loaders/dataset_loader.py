import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import os
import uuid
import csv
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from components.utils.export_utils import Export


class Annotation():
    def __init__(self, translation: list, size: list, num_points: int, eval_filename: str, mins: list, maxs: list, track_id: int):
        self.translation = translation
        self.size = size
        self.num_points = num_points
        self.eval_filename = eval_filename
        self.token = str(uuid.uuid4())
        self.mins = mins
        self.maxs = maxs
        self.track_id = track_id
        if self.size[2] > 1.5 and (self.size[1] < 0.3 or self.size[2] < 0.3):
            self.occluded = True
        else:
            self.occluded = False

    def getAsDict(self) -> dict:
        annotation = {}
        annotation["translation"] = self.translation
        annotation["size"] = self.size
        annotation["token"] = self.token
        annotation["num_points"] = self.num_points
        annotation["eval_filename"] = self.eval_filename
        annotation["occluded"] = self.occluded
        annotation["mins"] = self.mins
        annotation["maxs"] = self.maxs
        annotation["track_id"] = self.track_id
        annotation["class"] = "Person"

        return annotation

class LinkedDataHandler:
    def __init__(self, root_dir: str, anno_file = None, load_dataset = True, split_file: str = "split.json"):
        self.root_dir = Path(root_dir)
        self.dataset_json_path = self.root_dir / "dataset.json"

        if not self.dataset_json_path.exists():
            raise FileNotFoundError(f"Missing dataset.json in {self.root_dir}")

        with open(self.dataset_json_path, "r") as f:
            self.dataset_info = json.load(f)

        self.subsets: Dict[str, Dict[str, Any]] = {}
        self.train = []
        self.test = []
        self.val = []
        self.anno_file = anno_file
        self.split_file = split_file
        self.files_to_remove = []
        if load_dataset:
            self._load_all_subsets()

    def _load_all_subsets(self):
        for entry in self.dataset_info:
            self._load_subset(entry["subset_folder"])

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            print(f"Missing: {path}")
            return []
        with open(path, "r") as f:
            return json.load(f)

    def _load_subset(self, subset_name: str):
        if subset_name in self.subsets:
            return self.subsets[subset_name]

        subset_entry = next((e for e in self.dataset_info if e["subset_folder"] == subset_name), None)
        if not subset_entry:
            raise ValueError(f"Subset {subset_name} not found")

        subset_path = self.root_dir / subset_name
        samples = self._load_json(subset_path / "samples.json")
        links = self._load_json(subset_path / "links.json")
        calibrations = self._load_json(subset_path / "calibrations.json")
        if self.anno_file is None:
            annotations = self._load_json(subset_path / "annotations.json")
        else:
            annotations = self._load_json(subset_path / self.anno_file)
        for anno in annotations:
            if "class" not in anno.keys():
                anno["class"] = "Person"
        split = self._load_json(subset_path / self.split_file)

        sample_index = {s["token"]: s for s in samples}
        calib_index = {c["token"]: c["calibration"] for c in calibrations}
        annotation_index = {}
        for ann in annotations:
            annotation_index.setdefault(ann["sample_token"], []).append(ann)

        token_to_link = {}
        linked_data = []
        train_set = []
        test_set = []
        val_set = []
        for link in links:
            entry = dict(link)
            entry["samples"] = {}

            for key, token in link.items():
                if key.endswith("_token") and token:
                    sensor_type = key.replace("_token", "")
                    sample = sample_index.get(token)
                    if sample:
                        sample = dict(sample)
                        calib = calib_index.get(sample["calibrated_sensor_token"])
                        sample["calibration"] = calib
                        if sensor_type == "lidar":
                            sample["annotations"] = annotation_index.get(sample["token"], [])
                        entry["samples"][sensor_type] = sample

            token_to_link[entry["token"]] = entry
            linked_data.append(entry)
            if split != []:
                if link["token"] in split["train"]:
                    train_set.append(entry)
                    self.train.append((entry, subset_name))
                if link["token"] in split["test"]:
                    test_set.append(entry)
                    self.test.append((entry, subset_name))
                if link["token"] in split["val"]:
                    val_set.append(entry)
                    self.val.append((entry, subset_name))
            

        for entry in linked_data:
            entry["prev_link"] = token_to_link.get(entry.get("prev"))
            entry["next_link"] = token_to_link.get(entry.get("next"))

        subset_data = {
            "meta": subset_entry,
            "links": linked_data,
            "index": token_to_link,
            "raw": {
                "samples": samples,
                "calibrations": calibrations,
                "annotations": annotations,
            },
            "split": {
                "train": train_set,
                "test": test_set,
                "val": val_set
            }
        }

        self.subsets[subset_name] = subset_data
        self.setRequiredMetaFields(subset_name)
        return subset_data

    def list_subsets(self) -> List[str]:
        return list(self.subsets.keys())
    
    def list_all_subsets(self) -> List[str]:
        return [e["subset_folder"] for e in self.dataset_info]

    def get_subset(self, name: str) -> Dict[str, Any]:
        if name not in self.subsets:
            self._load_subset(name)
        return self.subsets[name]

    def get_links(self, name: str) -> List[Dict[str, Any]]:
        return self.get_subset(name)["links"]

    def get_link(self, name: str, token: str) -> Optional[Dict[str, Any]]:
        subset = self.get_subset(name)
        return subset["index"].get(token)
    
    def setRequiredMetaFields(self, subset_name: str):
        subset = self.subsets[subset_name]
        if "token" not in subset["meta"].keys():
            subset["meta"]["token"] = str(uuid.uuid4())
        if "bev_resolution" not in subset["meta"].keys():
            subset["meta"]["bev_resolution"] = 0.1
        if "allBBoxGrounded" not in subset["meta"].keys():
            subset["meta"]["allBBoxGrounded"] = False
        if "areaBBoxGrounded" not in subset["meta"].keys():
            subset["meta"]["areaBBoxGrounded"] = False
        if "enforceMinHeight" not in subset["meta"].keys():
            subset["meta"]["enforceMinHeight"] = False
        if "minHeight" not in subset["meta"].keys():
            subset["meta"]["minHeight"] = 1.9
        if "bev_min" not in subset["meta"].keys():
            subset["meta"]["bev_min"] = [-50,-50]
        if "bev_max" not in subset["meta"].keys():
            subset["meta"]["bev_max"] = [50,50]
        if "layer_polygons" not in subset["meta"].keys():
            subset["meta"]["layer_polygons"] = []
        if "clusters" not in subset["meta"].keys():
            subset["meta"]["clusters"] = ''
        if "minHeightDiff" not in subset["meta"].keys():
            subset["meta"]["minHeightDiff"] = 0.5
        if "standartSizeIncrease" not in subset["meta"].keys():
            subset["meta"]["standartSizeIncrease"] = 0.1
        if "standartSizeIncreaseInZ" not in subset["meta"].keys():
            subset["meta"]["standartSizeIncreaseInZ"] = False
        if "heightFromHighestPoint" not in subset["meta"].keys():
            subset["meta"]["heightFromHighestPoint"] = False
        if "takeMaxSize" not in subset["meta"].keys():
            subset["meta"]["takeMaxSize"] = False
        if "useValidArea" not in subset["meta"].keys():
            subset["meta"]["useValidArea"] = False

    def recomputeCombinedSplits(self):
        self.train.clear()
        self.test.clear()
        self.val.clear()
        for subset_name in self.subsets:
            for entry in self.subsets[subset_name]["split"]["train"]:
                self.train.append((self.get_link(subset_name, entry), subset_name))
            for entry in self.subsets[subset_name]["split"]["test"]:
                self.test.append((self.get_link(subset_name, entry), subset_name))
            for entry in self.subsets[subset_name]["split"]["val"]:
                self.val.append((self.get_link(subset_name, entry), subset_name))

    def __removeFile(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} has been deleted.")
        else:
            print(f"{file_path} does not exist.")

    def remove_link(self, link_dict: dict, subset: str):
        link = next((x for x in self.subsets[subset]["links"] if x['token'] == link_dict['token']), None)
        prev_link = next((x for x in self.subsets[subset]["links"] if x['token'] == link["prev"]), None)
        next_link = next((x for x in self.subsets[subset]["links"] if x['token'] == link["next"]), None)
        if link is None:
            return
        lidar = next((x for x in self.subsets[subset]["raw"]["samples"] if x['token'] == link['lidar_token']), None)
        thermal = next((x for x in self.subsets[subset]["raw"]["samples"] if x['token'] == link['thermal_token']), None)
        rgb = next((x for x in self.subsets[subset]["raw"]["samples"] if x['token'] == link['rgb_token']), None)
        annotations = self.subsets[subset]["raw"]["annotations"]
        while (annotation := next((x for x in annotations if x['sample_token'] == rgb['token']), None)):
            annotations.remove(annotation)
        while (annotation := next((x for x in annotations if x['sample_token'] == lidar['token']), None)):
            annotations.remove(annotation)
        while (annotation := next((x for x in annotations if x['sample_token'] == thermal['token']), None)):
            annotations.remove(annotation)

        self.files_to_remove.append(str(self.root_dir) +"/"+ subset+"/samples"+thermal['filename'])
        self.files_to_remove.append(str(self.root_dir) +"/"+ subset+"/samples"+rgb['filename'])
        self.files_to_remove.append(str(self.root_dir) +"/"+ subset+"/samples"+lidar['filename'])
        if prev_link is None:
            next_link["prev"] = ''    
        else:
            if next_link is not None:
                next_link["prev"] = prev_link["token"]
        if next_link is None:
            prev_link["next"] = ''
        else:
            if prev_link is not None:
                prev_link["next"] = next_link["token"]

        self.subsets[subset]["raw"]["samples"].remove(lidar)
        self.subsets[subset]["raw"]["samples"].remove(thermal)
        self.subsets[subset]["raw"]["samples"].remove(rgb)
        self.subsets[subset]["links"].remove(link)
        return None

    def link_exists(self, link_token, subset):
        for link in self.subsets[subset]["links"]:
            if link["token"] == link_token:
                return True
        return False
    
    def clear_splits(self):
        for subset in self.subsets:
            self.subsets[subset]["split"] = {'train':[], 'test':[], 'val':[]}
        self.train = []
        self.test = []
        self.val = []

    def createSplits(self, train_size: float, test_size: float, val_size: float):
        from sklearn.model_selection import train_test_split
        assert(train_size + test_size + val_size == 1.0)
        for subset in self.list_subsets():
            links = self.get_links(subset)
            train, tmp = train_test_split(links, test_size=(test_size + val_size), train_size=train_size)
            if val_size > 0.0:
                test, val = train_test_split(tmp, train_size=test_size, test_size=val_size)
            else:
                test = tmp
                val = []
    
            train_list = []
            for item in train:
                train_list.append(item["token"])
            test_list = []
            for item in test:
                test_list.append(item["token"])
            val_list = []
            for item in val:
                val_list.append(item["token"])

            self.subsets[subset]["split"]["train"] = train_list
            self.subsets[subset]["split"]["test"] = test_list
            self.subsets[subset]["split"]["val"] = val_list

    def getAttributesAndValues(self, attribute_names = ["location", "action", "weather", "visibilty", "isOutside", "other"]):
        attributes = {}
        for attribute_name in attribute_names:
            attributes[attribute_name] = []
        for subset in self.list_subsets():
            for attribute_name in attribute_names:
                if isinstance(self.subsets[subset]["meta"][attribute_name], List):
                    for a in self.subsets[subset]["meta"][attribute_name]:
                        if a not in attributes[attribute_name]:
                            attributes[attribute_name].append(a)
                else:
                    if self.subsets[subset]["meta"][attribute_name] not in attributes[attribute_name]:
                        attributes[attribute_name].append(self.subsets[subset]["meta"][attribute_name])

        return attributes

    def getAnnotationInCameraFrame(self, annotation: dict, link: dict, camera: str = "rgb"):
        assert camera == "thermal" or camera == "rgb"
        K = np.array(link['samples'][camera]['calibration']["camera_intrinisc"])
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        # --- Extrinsics ---
        q_cam = link['samples'][camera]["calibration"]["rotation"]  # [x,y,z,w]
        t_cam = np.array(link['samples'][camera]["calibration"]["translation"])
        R_cam2world = R.from_quat(q_cam).as_matrix()

        q_lidar = link['samples']['lidar']["calibration"]["rotation"]
        t_lidar = np.array(link['samples']['lidar']["calibration"]["translation"])
        R_lidar2world = R.from_quat(q_lidar).as_matrix()
        center = np.array(annotation["translation"])
        size = np.array(annotation["size"])

        l, w, h = size
        # 8 axis-aligned corners
        corners = np.array([
            [ l/2,  w/2,  h/2],
            [ l/2, -w/2,  h/2],
            [-l/2, -w/2,  h/2],
            [-l/2,  w/2,  h/2],
            [ l/2,  w/2, -h/2],
            [ l/2, -w/2, -h/2],
            [-l/2, -w/2, -h/2],
            [-l/2,  w/2, -h/2],
        ])
        corners_lidar = corners + center
        # LiDAR → World
        corners_world = (R_lidar2world @ corners_lidar.T).T + t_lidar
        # World → Camera
        corners_cam = (R_cam2world.T @ (corners_world - t_cam).T).T
        # Project
        pts_2d = []
        for X in corners_cam:
            if X[2] <= 0:  # behind camera
                continue
            u = fx * (X[0]/X[2]) + cx
            v = fy * (X[1]/X[2]) + cy
            pts_2d.append((int(u), int(v)))
        return pts_2d

    def getSubsetsWithAttributes(self, attributes: dict):
        valid_subsets = []
        for subset in self.list_subsets():
            valid_attributes = {}
            for key, attribute in attributes.items():
                if isinstance(attribute, List):
                    in_list = False
                    for a in attribute:
                        if a in self.subsets[subset]["meta"][key]:
                            in_list = True
                            break
                    if in_list:
                        valid_attributes[key] = True
                    else:
                        valid_attributes[key] = False
                else:
                    inArray = False
                    if isinstance(self.subsets[subset]["meta"][key], bool):
                        if self.subsets[subset]["meta"][key] == attribute:
                            inArray = True
                    elif isinstance(self.subsets[subset]["meta"][key], List):
                        if attribute in self.subsets[subset]["meta"][key]:
                            inArray = True
                    else:
                        assert("Add new type")
                    if inArray:
                        valid_attributes[key] = True
                    else:
                        valid_attributes[key] = False
            valid = True
            for _, a in valid_attributes.items():
                valid = valid and a
            if valid:
                valid_subsets.append(subset)
        return valid_subsets

    def loadSplitOfSubset(self, subset: str):
        assert subset in self.list_subsets()
        self.train = self.subsets[subset]['split']['train']
        self.test = self.subsets[subset]['split']['test']
        self.val = self.subsets[subset]['split']['val']

    def createSplitsAttributes(self, train_size: float, test_size: float, val_size: float, attributes: dict, min_point_count: int):
        from sklearn.model_selection import train_test_split
        assert(train_size + test_size + val_size == 1.0)
        for subset in self.list_subsets():
            valid_attributes = {}
            for key, attribute in attributes.items():
                if isinstance(attribute, List):
                    in_list = False
                    for a in attribute:
                        if a in self.subsets[subset]["meta"][key]:
                            in_list = True
                            break
                    if in_list:
                        valid_attributes[key] = True
                    else:
                        valid_attributes[key] = False
                else:
                    inArray = False
                    if isinstance(self.subsets[subset]["meta"][key], bool):
                        if self.subsets[subset]["meta"][key] == attribute:
                            inArray = True
                    elif isinstance(self.subsets[subset]["meta"][key], List):
                        if attribute in self.subsets[subset]["meta"][key]:
                            inArray = True
                    else:
                        assert("Add new type")
                    if inArray:
                        valid_attributes[key] = True
                    else:
                        valid_attributes[key] = False
            valid = True
            for _, a in valid_attributes.items():
                valid = valid and a
            if valid:
                links = self.get_links(subset)
                valid_links = []
                for link in links:
                    all_anno_valid = True
                    for anno in link["samples"]["lidar"]["annotations"]:
                        if anno["num_points"] <= min_point_count:
                            all_anno_valid = False
                            break
                    if all_anno_valid:
                        valid_links.append(link)
                
                if test_size == 1.0:
                    train = []
                    val = []
                    test = valid_links
                elif train_size == 1.0:
                    val = []
                    test = []
                    train = valid_links
                elif val_size == 1.0:
                    train = []
                    test = []
                    val = valid_links
                else:
                    train, tmp = train_test_split(valid_links, test_size=(test_size + val_size), train_size=train_size)
                    if val_size > 0.0:
                        test, val = train_test_split(tmp, train_size=test_size, test_size=val_size)
                    else:
                        test = tmp
                        val = []
        
                train_list = []
                for item in train:
                    train_list.append(item["token"])
                test_list = []
                for item in test:
                    test_list.append(item["token"])
                val_list = []
                for item in val:
                    val_list.append(item["token"])

                self.subsets[subset]["split"]["train"] = train_list
                self.subsets[subset]["split"]["test"] = test_list
                self.subsets[subset]["split"]["val"] = val_list

    def createSplits(self, train_size: float, test_size: float, val_size: float, min_point_count: int):
        from sklearn.model_selection import train_test_split
        assert(train_size + test_size + val_size == 1.0)
        for subset in self.list_subsets():
            links = self.get_links(subset)
            valid_links = []
            for link in links:
                all_anno_valid = True
                for anno in link["samples"]["lidar"]["annotations"]:
                    if anno["num_points"] <= min_point_count:
                        all_anno_valid = False
                        break
                if all_anno_valid:
                    valid_links.append(link)
            train, tmp = train_test_split(valid_links, test_size=(test_size + val_size), train_size=train_size)
            if val_size > 0.0:
                test, val = train_test_split(tmp, train_size=test_size, test_size=val_size)
            else:
                test = tmp
                val = []
    
            train_list = []
            for item in train:
                train_list.append(item["token"])
            test_list = []
            for item in test:
                test_list.append(item["token"])
            val_list = []
            for item in val:
                val_list.append(item["token"])

            self.subsets[subset]["split"]["train"] = train_list
            self.subsets[subset]["split"]["test"] = test_list
            self.subsets[subset]["split"]["val"] = val_list
            
    def addAnnotation(self, link, annotation: Annotation):
        anno = annotation.getAsDict()
        anno["sample_token"] = link["samples"]["lidar"]["token"]
        link["samples"]["lidar"]["annotations"].append(anno)

    def rebuild_raw_data(self, subset_name: Optional[str] = None):
        if subset_name:
            subsets = [subset_name]
        else:
            subsets = list(self.subsets.keys())

        for name in subsets:
            subset = self.subsets[name]
            raw_samples = []
            raw_annos = []
            raw_calibs = []

            for link in subset["links"]:
                for sample in link["samples"].values():
                    raw_samples.append(sample)
                    if "annotations" in sample:
                        raw_annos.extend(sample["annotations"])
                    if "calibration" in sample:
                        raw_calibs.append({
                            "token": sample["calibrated_sensor_token"],
                            "calibration": sample["calibration"]
                        })

            subset["raw"]["samples"] = list({s["token"]: s for s in raw_samples}.values())
            subset["raw"]["annotations"] = list({a["token"]: a for a in raw_annos}.values()) if raw_annos else subset["raw"]["annotations"]
            subset["raw"]["calibrations"] = list({c["token"]: c for c in raw_calibs}.values())

    def fixLinks(self, subset_name: str) -> str:
        subset = self.subsets[subset_name]
        time_sort = []
        for idx, link in enumerate(subset["links"]):
            time_sort.append((link["samples"]["lidar"]["timestamp"], idx))
        time_sort = sorted(time_sort, key=lambda x: x[0])

        subset["links"][time_sort[0][1]]["prev"] = ''
        for _, value in enumerate(time_sort[:-1]):
            subset["links"][value[1]]["next"] = subset["links"][time_sort[value[1]+1][1]]["token"]
            subset["links"][time_sort[value[1]+1][1]]["prev"] = subset["links"][value[1]]["token"]
        subset["links"][time_sort[-1][1]]["next"] = ''
        return subset["links"][time_sort[0][1]]["token"]

    def save(self, subset_name: Optional[str] = None, link_file_name: Optional[str] = None, split_file_name: Optional[str] = None, sample_file_name: Optional[str] = None, anno_file_name: Optional[str] = None, calib_file_name: Optional[str] = None, dataset_file_name: Optional[str] = None):
        if subset_name:
            subsets_to_save = [subset_name]
        else:
            subsets_to_save = list(self.subsets.keys())

        self.rebuild_raw_data(subset_name)

        for name in subsets_to_save:
            subset = self.subsets[name]
            subset_path = self.root_dir / name

            token_to_sample = {s["token"]: s for s in subset["raw"]["samples"]}
            for link in subset["links"]:
                for sensor_type, sample in link["samples"].items():
                    token = sample["token"]
                    if token in token_to_sample:
                        token_to_sample[token].update(
                            {k: v for k, v in sample.items() if k not in ['calibration', 'annotation']}
                        )

            tmp_samples = []
            for sample in subset["raw"]["samples"]:
                tmp = {}
                for item_name, data in sample.items():
                    if item_name not in ['calibration', 'annotations']:
                        tmp[item_name] = data
                tmp_samples.append(tmp)

            new_links = [
                {k: v for k, v in link.items() if k not in ["samples", "prev_link", "next_link"]}
                for link in subset["links"]
            ]

            firstLink = None
            for link in subset["links"]:
                if link["prev"] == '':
                    if firstLink is None:
                        firstLink = link["token"]
                    else:
                        firstLink = self.fixLinks(name)
            
            for anno in subset["raw"]["annotations"]:
                if "class" not in anno.keys():
                    anno["class"] = "person"

            if firstLink is None:
                print("The subset" + name + "has to have a starting link")
                assert False
            
            for d in self.dataset_info:
                if d['subset_folder'] == name:
                    d['first_link_token'] = firstLink
            

            if link_file_name is None:
                with open(subset_path / "links.json", "w") as f:
                    json.dump(new_links, f, indent=2)
            else:
                with open(subset_path / link_file_name, "w") as f:
                    json.dump(new_links, f, indent=2)

            if sample_file_name is None:
                with open(subset_path / "samples.json", "w") as f:
                    json.dump(tmp_samples, f, indent=2)
            else:
                with open(subset_path / sample_file_name, "w") as f:
                    json.dump(tmp_samples, f, indent=2)
            if anno_file_name is None:
                with open(subset_path / "annotations.json", "w") as f:
                    json.dump(subset["raw"]["annotations"], f, indent=2)
            else:
                with open(subset_path / anno_file_name, "w") as f:
                    json.dump(subset["raw"]["annotations"], f, indent=2)
            if calib_file_name is None:
                with open(subset_path / "calibrations.json", "w") as f:
                    json.dump(subset["raw"]["calibrations"], f, indent=2)
            else:
                with open(subset_path / calib_file_name, "w") as f:
                    json.dump(subset["raw"]["calibrations"], f, indent=2)

            split_tokens = {
               "train": [entry["token"] if isinstance(entry, dict) else entry for entry in subset["split"]["train"]],
               "test": [entry["token"] if isinstance(entry, dict) else entry for entry in subset["split"]["test"]],
               "val": [entry["token"] if isinstance(entry, dict) else entry for entry in subset["split"]["val"]],
            }       
            if split_file_name is None:    
                with open(subset_path / self.split_file, "w") as f:    
                    json.dump(split_tokens, f, indent=2)
            else:
                with open(subset_path / split_file_name, "w") as f:    
                    json.dump(split_tokens, f, indent=2)

        if dataset_file_name is None:
            with open(self.dataset_json_path, "w") as f:
                json.dump(self.dataset_info, f, indent=2)
        else:
            with open(dataset_file_name, "w") as f:
                json.dump(self.dataset_info, f, indent=2)

        for file in self.files_to_remove:
            self.__removeFile(file)

        print("Saved all changes.")

    def __repr__(self):
        n = len(self.list_subsets())
        loaded = len(self.subsets)
        return f"<LinkedDataHandler: {n} subsets ({loaded} loaded)>"

    def exportAnnotationToCSV(self):
        for subset_name in self.list_subsets():
            subset = self.subsets[subset_name]
            with open(str(self.root_dir) + "/" + subset_name + "_annotations.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["lidar_file_name", "x","y","z","l","w","h","yaw","label","occluded"])
                for link in subset["links"]:
                    for ann in link["samples"]["lidar"]["annotations"]:
                        frame = link["samples"]["lidar"]["filename"][7:]
                        x,y,z = ann["translation"]
                        l,w,h = ann["size"]
                        yaw = 0  # add yaw if available

                        writer.writerow([frame, x, y, z, l, w, h, yaw, ann["class"], ann["occluded"]])
    
    def importAnnotationFromCSV(self, subset_name, path):
        if subset_name not in self.list_subsets():
            return
        subset = self.subsets[subset_name]
        lidar_path_to_link_idx = {}
        for idx, link in enumerate(subset["links"]):
            lidar_path_to_link_idx[link["samples"]["lidar"]["filename"][7:]] = idx
            link["samples"]["lidar"]["annotations"].clear()
        with open(path, mode ='r')as file:
            csvFile = csv.reader(file)
            heading = next(csvFile)
            for lines in csvFile:
                tmp = Annotation([float(lines[1]), float(lines[2]), float(lines[3])], [float(lines[4]), float(lines[5]), float(lines[6])], -1, "", [], [], -1)
                d = tmp.getAsDict()
                d["sample_token"] =  subset["links"][lidar_path_to_link_idx[lines[0]]]["samples"]["lidar"]["token"]
                subset["links"][lidar_path_to_link_idx[lines[0]]]["samples"]["lidar"]["annotations"].append(d)

    def importAnnotationFromJson(self, subset_name, path):
        if subset_name not in self.list_subsets():
            return
        subset = self.subsets[subset_name]
        lidar_path_to_link_idx = {}
        for idx, link in enumerate(subset["links"]):
            lidar_path_to_link_idx[link["samples"]["lidar"]["filename"][7:]] = idx
            link["samples"]["lidar"]["annotations"].clear()
        with open(path, mode ='r')as file:
            data = json.load(file)
            for sample in data['items']:
                dataset_link = subset["links"][lidar_path_to_link_idx[sample["point_cloud"]["path"]]]
                pcd = o3d.io.read_point_cloud(str(self.root_dir)  + "/" + subset_name + "/samples" + dataset_link["samples"]["lidar"]["filename"])
                points = np.asarray(pcd.points)
                for anno in sample['annotations']:
                    if "track_id" in anno["attributes"].keys():
                        tmp = Annotation(anno["position"], anno["scale"], 0, "", [], [], anno["attributes"]["track_id"])
                    else:
                        tmp = Annotation(anno["position"], anno["scale"], 0, "", [], [], -1)
                    tmp = tmp.getAsDict()
                    tmp["sample_token"] = dataset_link["samples"]["lidar"]["token"]
                    center = np.array(tmp["translation"])
                    size = np.array(tmp["size"])
                    half_size = size / 2.0
                    bbox_min = center - half_size
                    bbox_max = center + half_size 
                    mask = (
                        (points[:, 0] >= bbox_min[0]) & (points[:, 0] <= bbox_max[0]) &
                        (points[:, 1] >= bbox_min[1]) & (points[:, 1] <= bbox_max[1]) &
                        (points[:, 2] >= bbox_min[2]) & (points[:, 2] <= bbox_max[2])
                    )

                    points_inside_bbox = points[mask]
                    #tmp.occluded = anno["attributes"]["occluded"]
                    tmp["num_points"] = len(points_inside_bbox)
                    if tmp["num_points"] > 0:
                        tmp["mins"] = points_inside_bbox.min(axis=0).tolist()
                        tmp["maxs"] = points_inside_bbox.min(axis=0).tolist()
                        

                        dataset_link["samples"]["lidar"]["annotations"].append(tmp)
                    
    def setAnnotationPointCount(self):
        for subset_name in self.list_subsets():
            subset = self.subsets[subset_name]
            for idx, link in enumerate(subset["links"]):
                sample = link["samples"]["lidar"]
                pcd = o3d.io.read_point_cloud(str(self.root_dir)  + "/" + subset_name + "/samples" + sample["filename"])
                points = np.asarray(pcd.points)
                for anno in sample['annotations']:
                    center = np.array(anno["translation"])
                    size = np.array(anno["size"])
                    half_size = size / 2.0
                    bbox_min = center - half_size
                    bbox_max = center + half_size 
                    mask = (
                        (points[:, 0] >= bbox_min[0]) & (points[:, 0] <= bbox_max[0]) &
                        (points[:, 1] >= bbox_min[1]) & (points[:, 1] <= bbox_max[1]) &
                        (points[:, 2] >= bbox_min[2]) & (points[:, 2] <= bbox_max[2])
                    )
                    points_inside_bbox = points[mask]
                    anno["num_points"] = len(points_inside_bbox)

                    if anno["num_points"] > 0:
                        anno["mins"] = points_inside_bbox.min(axis=0).tolist()
                        anno["maxs"] = points_inside_bbox.max(axis=0).tolist()
                    else:
                        anno["maxs"] = []
                        anno["mins"] = []
                        print("fix subset:", subset_name, "lidar token:", sample["token"], "anno_token:", anno["token"], "id:", idx)

    def exportAnnotations(self, keyFrameSteps: int = 10):
        for subset_name in self.list_subsets():
            subset = self.subsets[subset_name]
            export = {
                "info": {},
                "categories": {
                    "label": {
                        "labels": [
                            {
                                "name": "person",
                                "parent": "",
                                "attributes": []
                            }
                        ],
                        "label_groups": [],
                        "attributes": ["occluded"]
                    },
                    "points": {
                        "items": []
                    }
                },
                "items": []
            }

            items = []
            frame_id = 0
            item_id = 0
            for link in subset["links"]:
                item = {
                    "id": link["samples"]["lidar"]["filename"][7:-4],
                    "annotations": [],
                    "attr": {"frame": frame_id},
                    "point_cloud": {"path": link["samples"]["lidar"]["filename"][7:]},
                }
                for i, anno in enumerate(link["samples"]["lidar"]["annotations"]):
                    keyFrame = False
                    if frame_id%keyFrameSteps == 0:
                        keyFrame = True
                    ann_size = anno["size"]
                    ann = {
                        "id": i,
                        "type": "cuboid_3d",
                        "attributes": {
                            "occluded": anno.get("occluded", False),
                            "track_id": anno.get("track_id", 0),
                            "keyframe": keyFrame,
                        },
                        "group": 0,
                        "label_id": 0,
                        "position": anno["translation"],
                        "rotation": [0, 0, 0],
                        "scale": Export.swap_cvat_dimensions(ann_size),
                    }
                    item["annotations"].append(ann)
                items.append(item)
                frame_id += 1
                item_id += 1

            # Reverse frame order for CVAT export to match external tools.
            export["items"] = Export.reverse_frame_order(items)

            with open(str(self.root_dir) + "/" + subset_name + "_cvat.json", "w") as f:
                f.write(json.dumps(export, indent=2))




