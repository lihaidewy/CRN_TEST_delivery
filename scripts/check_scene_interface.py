#!/usr/bin/env python3
"""Fast CPU-only regression checks for dynamic scene calculations."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scene_config import SceneConfig, load_scene_config
from scripts.data.scene_preprocess_interface import radar_point_to_bev


def main() -> None:
    path = PROJECT_ROOT / 'configs' / 'scene_s3m_example.json'
    scene = load_scene_config(path)
    derived = scene.derived
    assert derived['bev_shape'] == [200, 480]
    assert derived['grid_size'] == [480, 200, 1]
    assert derived['depth_bins'] == 120
    assert derived['pv_output_shape'] == [240, 176]

    point = {'range_m': 50.0, 'angle': 0.0, 'radial_speed_mps': 10.0, 'snr': 1.0}
    xyz, velocity = radar_point_to_bev(point, scene.raw)
    assert np.allclose(xyz, [40.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(velocity, [10.0, 0.0], atol=1e-6)

    invalid = json.loads(path.read_text(encoding='utf-8'))
    invalid['bev']['x_bound_m'] = [0.0, 180.0]  # 450 cells: known FPN mismatch case.
    try:
        SceneConfig(copy.deepcopy(invalid), path)
    except ValueError as error:
        assert 'divisible by 8' in str(error)
    else:
        raise AssertionError('Invalid 450-cell BEV grid was not rejected.')

    print('Dynamic scene interface calculations: PASS')


if __name__ == '__main__':
    main()
