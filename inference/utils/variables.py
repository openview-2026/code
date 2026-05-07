from pydantic import BaseModel
from typing import List, Dict, Tuple

class ViewMeta(BaseModel):
    view_id: str
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    fov_x_deg: float
    fov_y_deg: float
    fov_diag_deg: float
    aspect_ratio: float
    image_path: str
    left_neighbor_view: str = "Edge"
    right_neighbor_view: str = "Edge"
    top_neighbor_view: str = "Edge"
    bottom_neighbor_view: str = "Edge"

class ViewCaption(BaseModel):
    view_id: str
    uv_meta: Dict[str, float | str] = {
        "u_norm": 0.0, "v_norm": 0.0, 
        "fov_x_deg":0.0, "fov_y_deg":0.0, 
        "fov_diag_deg": 0.0, "aspect_ratio": "1:1",
        "left_neighbor_view": "Edge",
        "right_neighbor_view": "Edge",
        "top_neighbor_view": "Edge",
        "bottom_neighbor_view": "Edge",
        }
    caption: str
    objects: List[str] = []
    spatial_facts: List[str] = []

class PanoSummary(BaseModel):
    pano_id: str
    pano_size: List[int]
    pano_path: str
    summary: str
    label: str
    outdoor: str = "Not Given"