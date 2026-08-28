from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_NAME = "ballistic_soccer"
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / EXAMPLE_NAME


class SimulationError(RuntimeError):
    pass


def _imports(gl_backend: str) -> tuple[Any, Any]:
    os.environ.setdefault("MUJOCO_GL", gl_backend)
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        raise SimulationError(
            "MuJoCo and NumPy are required; install requirements/simulation.txt"
        ) from exc
    return mujoco, np


def load_example() -> tuple[Path, dict[str, Any]]:
    scene = json.loads((EXAMPLE_ROOT / "scene.json").read_text(encoding="utf-8"))
    return EXAMPLE_ROOT, scene


def _coordinate_matrix(np: Any) -> Any:
    # MuJoCo z-up -> released scene y-up: (x, y, z) -> (x, z, -y).
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )


def _project_to_mujoco(value: Any, np: Any) -> Any:
    return np.einsum("ij,...j->...i", _coordinate_matrix(np).T, np.asarray(value, dtype=float))


def _quaternion_xyzw_to_matrix(value: Any, np: Any) -> Any:
    quaternion = np.asarray(value, dtype=np.float64)
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion_xyzw(matrix: Any, np: Any) -> Any:
    value = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(value))
    if trace > 0.0:
        root = math.sqrt(trace + 1.0) * 2.0
        result = np.asarray(
            [
                (value[2, 1] - value[1, 2]) / root,
                (value[0, 2] - value[2, 0]) / root,
                (value[1, 0] - value[0, 1]) / root,
                0.25 * root,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(value)))
        if axis == 0:
            root = math.sqrt(max(0.0, 1 + value[0, 0] - value[1, 1] - value[2, 2])) * 2
            result = np.asarray(
                [
                    0.25 * root,
                    (value[0, 1] + value[1, 0]) / root,
                    (value[0, 2] + value[2, 0]) / root,
                    (value[2, 1] - value[1, 2]) / root,
                ]
            )
        elif axis == 1:
            root = math.sqrt(max(0.0, 1 + value[1, 1] - value[0, 0] - value[2, 2])) * 2
            result = np.asarray(
                [
                    (value[0, 1] + value[1, 0]) / root,
                    0.25 * root,
                    (value[1, 2] + value[2, 1]) / root,
                    (value[0, 2] - value[2, 0]) / root,
                ]
            )
        else:
            root = math.sqrt(max(0.0, 1 + value[2, 2] - value[0, 0] - value[1, 1])) * 2
            result = np.asarray(
                [
                    (value[0, 2] + value[2, 0]) / root,
                    (value[1, 2] + value[2, 1]) / root,
                    0.25 * root,
                    (value[1, 0] - value[0, 1]) / root,
                ]
            )
    result /= np.linalg.norm(result)
    return -result if result[3] < 0 else result


def _project_quaternion_to_mujoco(value: Any, np: Any) -> Any:
    coordinate = _coordinate_matrix(np)
    rotation = _quaternion_xyzw_to_matrix(value, np)
    return _matrix_to_quaternion_xyzw(coordinate.T @ rotation @ coordinate, np)


def _project_quaternion_from_mujoco(value_wxyz: Any, np: Any) -> Any:
    coordinate = _coordinate_matrix(np)
    q = np.asarray(value_wxyz, dtype=np.float64)
    rotation = _quaternion_xyzw_to_matrix(np.r_[q[1:], q[0]], np)
    return _matrix_to_quaternion_xyzw(coordinate @ rotation @ coordinate.T, np)


def _timeline(scene: Mapping[str, Any], np: Any) -> Any:
    binding = scene.get("camera", {}).get("binding", {})
    points = binding.get("pts_seconds") if isinstance(binding, Mapping) else None
    if isinstance(points, list) and points:
        result = np.asarray(points, dtype=np.float64)
    else:
        timeline = scene.get("timeline", {})
        frame_count = int(timeline["frame_count"])
        fps = float(timeline["fps"])
        result = np.arange(frame_count, dtype=np.float64) / fps
    return result


def _dynamic_rows(scene: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in scene["objects"] if row.get("role") != "static_rigid"]


def _initialize_state(
    scene: Mapping[str, Any], model: Any, data: Any, mujoco: Any, np: Any, timestamps: Any
) -> list[dict[str, Any]]:
    angular_velocities: list[tuple[int, int, Any]] = []
    delayed_activation: list[dict[str, Any]] = []
    for row in _dynamic_rows(scene):
        identifier = str(row["id"])
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, identifier)
        if body_id < 0 or int(model.body_jntnum[body_id]) < 1:
            raise SimulationError(f"dynamic body is missing from MJCF: {identifier}")
        joint_id = int(model.body_jntadr[body_id])
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise SimulationError(f"dynamic body must use a free joint: {identifier}")
        qpos = int(model.jnt_qposadr[joint_id])
        dof = int(model.jnt_dofadr[joint_id])
        state = row.get("initial_state", {})
        position = np.asarray(state.get("position"), dtype=np.float64)
        rotation = np.asarray(state.get("rotation_xyzw", [0, 0, 0, 1]), dtype=np.float64)
        velocity = np.asarray(state.get("linear_velocity", [0, 0, 0]), dtype=np.float64)
        angular = np.asarray(state.get("angular_velocity", [0, 0, 0]), dtype=np.float64)
        quaternion = _project_quaternion_to_mujoco(rotation, np)
        data.qpos[qpos : qpos + 3] = _project_to_mujoco(position, np)
        data.qpos[qpos + 3 : qpos + 7] = np.r_[quaternion[3], quaternion[:3]]
        data.qvel[dof : dof + 3] = _project_to_mujoco(velocity, np)
        angular_velocities.append((body_id, dof, _project_to_mujoco(angular, np)))

        seed = row.get("physics", {}).get("fit", {}).get("initial_state_seed", {})
        activation_frame = seed.get("collision_activation_frame_index") if isinstance(seed, Mapping) else None
        activation_source = seed.get("collision_activation_source") if isinstance(seed, Mapping) else None
        accepted_sources = {
            "first_valid_track_after_offscreen_ballistic_extrapolation",
            "first_valid_track_clear_of_support_after_upward_crossing",
        }
        if isinstance(activation_frame, int) and 0 < activation_frame < len(timestamps) and activation_source in accepted_sources:
            start = int(model.body_geomadr[body_id])
            count = int(model.body_geomnum[body_id])
            geom_ids = np.arange(start, start + count, dtype=np.int32)
            collidable = geom_ids[
                (np.asarray(model.geom_contype[geom_ids]) != 0)
                | (np.asarray(model.geom_conaffinity[geom_ids]) != 0)
            ]
            if len(collidable):
                delayed_activation.append(
                    {
                        "time_seconds": float(timestamps[activation_frame]),
                        "geom_ids": collidable,
                        "contype": np.asarray(model.geom_contype[collidable], dtype=np.int32).copy(),
                        "conaffinity": np.asarray(model.geom_conaffinity[collidable], dtype=np.int32).copy(),
                        "activated": False,
                    }
                )
                model.geom_contype[collidable] = 0
                model.geom_conaffinity[collidable] = 0

    mujoco.mj_forward(model, data)
    for body_id, dof, angular_world in angular_velocities:
        world_from_body = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        data.qvel[dof + 3 : dof + 6] = world_from_body.T @ angular_world
    mujoco.mj_forward(model, data)
    return delayed_activation


def _prepare_events(scene: Mapping[str, Any], model: Any, mujoco: Any, np: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in scene.get("external_wrenches", []):
        event_id = str(raw["id"])
        object_id = str(raw["object_id"])
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
        if body_id < 0:
            raise SimulationError(f"external wrench targets missing body: {object_id}")
        duration = float(raw["duration_seconds"])
        start = float(raw["start_seconds"])
        impulse = np.asarray(raw["impulse_world"], dtype=np.float64)
        torque = np.asarray(raw.get("torque_impulse_world", [0, 0, 0]), dtype=np.float64)
        impulse_class = str(raw.get("impulse_class", "external"))
        impulse_semantics = raw.get("impulse_semantics")
        native_contact = impulse_class in {
            "boundary_contact_constraint",
            "support_contact_constraint",
        }
        raw_support_normal = raw.get("support_normal_world")
        support_normal = (
            _project_to_mujoco(raw_support_normal, np)
            if isinstance(raw_support_normal, list) and len(raw_support_normal) == 3
            else None
        )
        contact_geom_ids: tuple[int, int] | None = None
        if impulse_class == "boundary_contact_constraint":
            boundary = str(raw.get("boundary", ""))
            first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"sdk_table_rail_{boundary}")
            second = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{object_id}_collision")
            if first >= 0 and second >= 0:
                contact_geom_ids = (int(first), int(second))
        elif impulse_class == "support_contact_constraint":
            support_id = str(raw.get("support_object_id", ""))
            first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{support_id}_collision")
            second = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{object_id}_collision")
            if first >= 0 and second >= 0:
                contact_geom_ids = (int(first), int(second))
        events.append(
            {
                "id": event_id,
                "kind": str(raw.get("kind", "")),
                "body_id": int(body_id),
                "object_id": object_id,
                "start_seconds": start,
                "end_seconds": start + duration,
                "duration_seconds": duration,
                "impulse": _project_to_mujoco(impulse, np),
                "torque": _project_to_mujoco(torque, np),
                "native_contact": native_contact,
                "contact_geom_ids": contact_geom_ids,
                "contact_tolerance": float(raw.get("contact_time_tolerance_seconds", duration)),
                "application_phase": str(raw.get("application_phase", "pre_native_contact")),
                "impulse_semantics": impulse_semantics,
                "support_normal": support_normal,
            }
        )
    return events


def _contact_point(data: Any, event: Mapping[str, Any], np: Any) -> Any | None:
    pair = event.get("contact_geom_ids")
    if not pair:
        return None
    expected = frozenset((int(pair[0]), int(pair[1])))
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if frozenset((int(contact.geom1), int(contact.geom2))) == expected:
            return np.asarray(contact.pos, dtype=np.float64).copy()
    return None


def _apply_force_windows(
    data: Any, events: Sequence[Mapping[str, Any]], start: float, step: float, np: Any
) -> None:
    data.xfrc_applied[:] = 0.0
    end = start + step
    for event in events:
        if event["kind"] != "force_window":
            continue
        overlap = max(0.0, min(end, float(event["end_seconds"])) - max(start, float(event["start_seconds"])))
        if overlap <= 0:
            continue
        scale = overlap / (float(event["duration_seconds"]) * step)
        body_id = int(event["body_id"])
        data.xfrc_applied[body_id, :3] += np.asarray(event["impulse"]) * scale
        data.xfrc_applied[body_id, 3:] += np.asarray(event["torque"]) * scale


def _apply_impulses(
    model: Any,
    data: Any,
    mujoco: Any,
    np: Any,
    events: Sequence[Mapping[str, Any]],
    now: float,
    applied: set[str],
    phase: str,
) -> None:
    due: list[tuple[Mapping[str, Any], Any | None]] = []
    for event in events:
        if event["kind"] not in {"impulse", "contact_impulse"} or event["id"] in applied:
            continue
        if event["application_phase"] != phase:
            continue
        contact_point = None
        if event["native_contact"]:
            start = float(event["start_seconds"])
            tolerance = float(event["contact_tolerance"])
            if not start - tolerance <= now <= start + tolerance:
                continue
            contact_point = _contact_point(data, event, np)
            if contact_point is None:
                continue
        elif float(event["start_seconds"]) > now + 1e-12:
            continue
        due.append((event, contact_point))
    if not due:
        return

    def resolved_linear_impulse(event: Mapping[str, Any]) -> Any:
        target = np.asarray(event["impulse"], dtype=np.float64)
        semantics = event.get("impulse_semantics")
        if semantics not in {
            "fitted_outgoing_normal_momentum",
            "fitted_outgoing_contact_momentum",
        }:
            return target
        body_id = int(event["body_id"])
        joint_id = int(model.body_jntadr[body_id])
        dof = int(model.jnt_dofadr[joint_id])
        current = float(model.body_mass[body_id]) * np.asarray(
            data.qvel[dof : dof + 3], dtype=np.float64
        )
        if semantics == "fitted_outgoing_contact_momentum":
            return target - current
        normal = np.asarray(event.get("support_normal"), dtype=np.float64)
        return (float(np.dot(target, normal)) - float(np.dot(current, normal))) * normal

    mujoco.mj_forward(model, data)
    generalized = np.zeros(model.nv, dtype=np.float64)
    for event, point in due:
        body_id = int(event["body_id"])
        mujoco.mj_applyFT(
            model,
            data,
            resolved_linear_impulse(event),
            np.asarray(event["torque"], dtype=np.float64),
            point if point is not None else np.asarray(data.xipos[body_id], dtype=np.float64),
            body_id,
            generalized,
        )
    mass_matrix = np.empty((model.nv, model.nv), dtype=np.float64)
    try:
        # MuJoCo 3.11 accepts MjData directly; older 3.x bindings expose qM.
        mujoco.mj_fullM(model, data, mass_matrix)
    except TypeError:
        mujoco.mj_fullM(model, mass_matrix, data.qM)
    data.qvel[:] += np.linalg.solve(mass_matrix, generalized)
    applied.update(str(event["id"]) for event, _ in due)


def simulate_case(case_root: Path, scene: Mapping[str, Any], gl_backend: str) -> dict[str, Any]:
    mujoco, np = _imports(gl_backend)
    xml_path = case_root / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    timestamps = _timeline(scene, np)
    moving = _dynamic_rows(scene)
    delayed = _initialize_state(scene, model, data, mujoco, np, timestamps)
    events = _prepare_events(scene, model, mujoco, np)

    object_ids = [str(row["id"]) for row in moving]
    positions = np.empty((len(timestamps), len(moving), 3), dtype=np.float64)
    rotations = np.empty((len(timestamps), len(moving), 4), dtype=np.float64)
    linear_velocities = np.empty_like(positions)
    angular_velocities = np.empty_like(positions)
    qpos_frames = np.empty((len(timestamps), model.nq), dtype=np.float64)

    def activate(timestamp: float) -> None:
        for item in delayed:
            if item["activated"] or timestamp < item["time_seconds"] - 1e-12:
                continue
            geom_ids = item["geom_ids"]
            model.geom_contype[geom_ids] = item["contype"]
            model.geom_conaffinity[geom_ids] = item["conaffinity"]
            item["activated"] = True

    def capture(frame: int) -> None:
        qpos_frames[frame] = data.qpos
        coordinate = _coordinate_matrix(np)
        for index, identifier in enumerate(object_ids):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, identifier)
            positions[frame, index] = coordinate @ np.asarray(data.xpos[body_id])
            rotations[frame, index] = _project_quaternion_from_mujoco(data.xquat[body_id], np)
            joint_id = int(model.body_jntadr[body_id])
            dof = int(model.jnt_dofadr[joint_id])
            linear_velocities[frame, index] = coordinate @ np.asarray(data.qvel[dof : dof + 3])
            world_from_body = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
            angular_velocities[frame, index] = coordinate @ (
                world_from_body @ np.asarray(data.qvel[dof + 3 : dof + 6])
            )

    elapsed = float(timestamps[0])
    base_step = float(model.opt.timestep)
    applied: set[str] = set()
    for frame, timestamp_raw in enumerate(timestamps):
        timestamp = float(timestamp_raw)
        while elapsed < timestamp - 1e-12:
            activate(elapsed)
            _apply_impulses(model, data, mujoco, np, events, elapsed, applied, "pre_native_contact")
            step = min(base_step, timestamp - elapsed)
            scheduled = [
                float(event["start_seconds"])
                for event in events
                if event["kind"] in {"impulse", "contact_impulse"}
                and event["id"] not in applied
                and not event["native_contact"]
                and float(event["start_seconds"]) > elapsed + 1e-12
            ]
            if scheduled:
                step = min(step, min(scheduled) - elapsed)
            activation_times = [
                float(item["time_seconds"])
                for item in delayed
                if not item["activated"] and float(item["time_seconds"]) > elapsed + 1e-12
            ]
            if activation_times:
                step = min(step, min(activation_times) - elapsed)
            if step <= 1e-12:
                raise SimulationError("event scheduler failed to make forward progress")
            model.opt.timestep = step
            _apply_force_windows(data, events, elapsed, step, np)
            mujoco.mj_step(model, data)
            _apply_impulses(model, data, mujoco, np, events, elapsed + step, applied, "post_native_contact")
            elapsed += step
        data.xfrc_applied[:] = 0.0
        activate(timestamp)
        mujoco.mj_forward(model, data)
        _apply_impulses(model, data, mujoco, np, events, timestamp, applied, "pre_native_contact")
        capture(frame)
    model.opt.timestep = base_step
    return {
        "model": model,
        "timestamps": timestamps,
        "object_ids": object_ids,
        "roles": [str(row.get("role", "")) for row in moving],
        "semantic_labels": [str(row.get("semantic_label", row["id"])) for row in moving],
        "extent": np.asarray([row.get("extent", [0.0, 0.0, 0.0]) for row in moving]),
        "position": positions,
        "rotation_xyzw": rotations,
        "linear_velocity": linear_velocities,
        "angular_velocity": angular_velocities,
        "qpos": qpos_frames,
        "applied_event_ids": sorted(applied),
    }


def save_trajectory(path: Path, rollout: Mapping[str, Any]) -> None:
    _, np = _imports(os.environ.get("MUJOCO_GL", "egl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pts_seconds=rollout["timestamps"],
        object_ids=np.asarray(rollout["object_ids"], dtype="U128"),
        roles=np.asarray(rollout["roles"], dtype="U32"),
        semantic_labels=np.asarray(rollout["semantic_labels"], dtype="U128"),
        position=rollout["position"],
        rotation_xyzw=rollout["rotation_xyzw"],
        linear_velocity=rollout["linear_velocity"],
        angular_velocity=rollout["angular_velocity"],
        extent=rollout["extent"],
    )


def _camera_for_rollout(
    scene: Mapping[str, Any],
    rollout: Mapping[str, Any],
    mujoco: Any,
    np: Any,
    azimuth_offset: float,
    elevation_offset: float,
) -> Any:
    position_project = np.asarray(rollout["position"], dtype=np.float64)
    flattened = position_project.reshape(-1, 3)
    center_project = np.median(flattened, axis=0)
    distances = np.linalg.norm(position_project - center_project, axis=-1)
    radius = max(float(np.quantile(distances, 0.95)), 1.0)

    binding = scene.get("camera", {}).get("binding", {})
    raw = np.asarray(
        binding.get("T_world_camera_first") if isinstance(binding, Mapping) else None,
        dtype=np.float64,
    )
    if raw.shape == (4, 4) and np.isfinite(raw).all():
        eye_project = raw[:3, 3]
        forward_project = raw[:3, 2]
        norm = float(np.linalg.norm(forward_project))
    else:
        norm = 0.0

    if norm > 1e-9:
        forward_project = forward_project / norm
        target_project = eye_project + forward_project * max(2.5 * radius, 2.0)
        eye = _project_to_mujoco(eye_project, np)
        lookat = _project_to_mujoco(target_project, np)
        delta = eye - lookat
        distance = max(float(np.linalg.norm(delta)), 2.0)
        azimuth = math.degrees(math.atan2(-delta[1], -delta[0]))
        elevation = math.degrees(
            math.asin(float(np.clip(-delta[2] / distance, -1.0, 1.0)))
        )
    else:
        lookat = _project_to_mujoco(center_project, np)
        distance = max(2.5 * radius, 2.0)
        azimuth = 90.0
        elevation = -20.0

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth + azimuth_offset
    camera.elevation = elevation + elevation_offset
    return camera


def _source_vertical_fov(scene: Mapping[str, Any], np: Any) -> float | None:
    binding = scene.get("camera", {}).get("binding", {})
    if not isinstance(binding, Mapping):
        return None
    intrinsics = np.asarray(binding.get("intrinsics_first"), dtype=np.float64)
    image_size = np.asarray(binding.get("image_size"), dtype=np.float64)
    if intrinsics.shape != (3, 3) or image_size.shape != (2,):
        return None
    focal_y = float(intrinsics[1, 1])
    image_height = float(image_size[0])
    if not math.isfinite(focal_y) or focal_y <= 0 or image_height <= 0:
        return None
    return math.degrees(2.0 * math.atan(image_height / (2.0 * focal_y)))


def render_video(
    path: Path,
    scene: Mapping[str, Any],
    rollout: Mapping[str, Any],
    *,
    width: int,
    height: int,
    fps: float,
    azimuth_offset: float,
    elevation_offset: float,
    gl_backend: str,
) -> None:
    mujoco, np = _imports(gl_backend)
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SimulationError(
            "imageio with ffmpeg support is required; install requirements/simulation.txt"
        ) from exc
    model = rollout["model"]
    data = mujoco.MjData(model)
    camera = _camera_for_rollout(
        scene, rollout, mujoco, np, azimuth_offset, elevation_offset
    )
    source_fov = _source_vertical_fov(scene, np)
    if source_fov is not None:
        model.vis.global_.fovy = source_fov
    model.vis.map.znear = min(float(model.vis.map.znear), 0.001)
    cast_shadow = getattr(model, "light_castshadow", None)
    if cast_shadow is not None:
        cast_shadow[:] = False
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        with imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            ffmpeg_log_level="error",
        ) as writer:
            for qpos in rollout["qpos"]:
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=options)
                writer.append_data(renderer.render())
    finally:
        renderer.close()


def run_example(args: argparse.Namespace) -> None:
    case_root, scene = load_example()
    output_dir = args.output_dir.resolve()
    trajectory_path = output_dir / f"{EXAMPLE_NAME}.npz"
    video_path = output_dir / f"{EXAMPLE_NAME}.mp4"
    print(f"[simulation] running {EXAMPLE_NAME}", flush=True)
    rollout = simulate_case(case_root, scene, args.mujoco_gl)
    save_trajectory(trajectory_path, rollout)
    if args.render:
        scene_fps = float(scene.get("timeline", {}).get("fps", 30.0))
        render_video(
            video_path,
            scene,
            rollout,
            width=args.width,
            height=args.height,
            fps=args.fps or scene_fps,
            azimuth_offset=args.azimuth_offset,
            elevation_offset=args.elevation_offset,
            gl_backend=args.mujoco_gl,
        )
        print(f"[simulation] video: {video_path}")
    print(f"[simulation] trajectory: {trajectory_path}")
    if rollout["applied_event_ids"]:
        print("[simulation] applied events: " + ", ".join(rollout["applied_event_ids"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bundled physics example")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "simulations")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--azimuth-offset", "--azimuth", type=float, default=0.0)
    parser.add_argument("--elevation-offset", "--elevation", type=float, default=0.0)
    parser.add_argument("--mujoco-gl", choices=("egl", "osmesa", "glfw"), default=os.environ.get("MUJOCO_GL", "egl"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.width <= 0 or args.height <= 0 or (args.fps is not None and args.fps <= 0):
            parser.error("render dimensions and fps must be positive")
        run_example(args)
        return 0
    except SimulationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
