"""Compare dual-antenna GPS and camera horizon pitch from a real-world ROS bag."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from scipy.spatial.transform import Rotation

from helpers.load_sbg_types import typestore as sbg_typestore
from helpers.params import RealWorldSBGConfigParams
from src.core.image_handler import ImageProcessor


IMAGE_TOPIC = "/oak_d_lr_1/rgb/image_raw"
GPS_TOPIC = "/sbg/gps_hdt"
EKF_TOPIC = "/sbg/ekf_quat"


def read_pitch(bag, max_images, gps_sign):
    config = RealWorldSBGConfigParams()
    camera = ImageProcessor(
        config.mtx, config.dist, config.img_width, config.img_height,
        camera_height=config.camera_height,
    )
    samples = {topic: {} for topic in (IMAGE_TOPIC, GPS_TOPIC, EKF_TOPIC)}
    image_count = 0

    with AnyReader([bag], default_typestore=sbg_typestore) as reader:
        connections = [c for c in reader.connections if c.topic in samples]
        missing = {IMAGE_TOPIC, GPS_TOPIC} - {c.topic for c in connections}
        if missing:
            raise ValueError(f"Bag is missing required topics: {', '.join(sorted(missing))}")

        for connection, _, rawdata in reader.messages(connections=connections):
            topic = connection.topic
            if topic == IMAGE_TOPIC:
                if max_images is not None and image_count == max_images:
                    break
                image_count += 1

            msg = reader.deserialize(rawdata, connection.msgtype)
            stamp = msg.header.stamp
            timestamp = stamp.sec + stamp.nanosec * 1e-9

            if topic == IMAGE_TOPIC:
                observation = camera.lsd_image_callback(message_to_cvimage(msg, "bgr8"))
                if not observation.valid:
                    continue
                pitch = np.rad2deg(observation.pitch_rad)
            elif topic == GPS_TOPIC:
                if msg.status & 0x3F:  # Low six bits are zero for SOL_COMPUTED.
                    continue
                pitch = gps_sign * msg.pitch
            else:
                quat = msg.quaternion
                pitch = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler(
                    "xyz", degrees=True
                )[1]

            if np.isfinite(pitch):
                samples[topic][timestamp] = pitch

    return tuple(
        np.array(sorted(samples[topic].items()), dtype=float).reshape(-1, 2)
        for topic in (GPS_TOPIC, IMAGE_TOPIC, EKF_TOPIC)
    )


def print_metrics(label, error):
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    print(f"{label} (n={error.size}): RMSE {rmse:.3f}°, MAE {mae:.3f}°, "
          f"bias {error.mean():+.3f}°, std dev {error.std():.3f}°")


def estimate_gps_bias(gps, ekf):
    if len(ekf) < 2:
        raise ValueError("GPS offset estimation requires at least two valid SBG EKF samples.")
    times, pitch = gps.T
    overlap = (times >= ekf[0, 0]) & (times <= ekf[-1, 0])
    if not overlap.any():
        raise ValueError("No valid GPS timestamps overlap SBG EKF pitch.")
    error = pitch[overlap] - np.interp(times[overlap], *ekf.T)
    return error.mean(), error


def compare_and_plot(gps, horizon, ekf, correct_gps_offset=False):
    if len(gps) == 0 or len(horizon) < 2:
        raise ValueError("Need at least one valid GPS fix and two valid horizon estimates.")

    times, gps_pitch = gps.T
    start = min(times[0], horizon[0, 0])
    overlap = (times >= horizon[0, 0]) & (times <= horizon[-1, 0])
    if not overlap.any():
        raise ValueError("No valid GPS timestamps overlap the horizon estimates.")

    horizon_at_gps = np.full(len(gps), np.nan)
    horizon_at_gps[overlap] = np.interp(times[overlap], *horizon.T)
    corrected_pitch = None
    if correct_gps_offset:
        bias, fit_error = estimate_gps_bias(gps, ekf)
        print(f"Estimated GPS bias relative to SBG EKF: {bias:+.4f}°")
        print(f"Shift to add to GPS pitch: {-bias:+.4f}°")
        print_metrics("GPS − SBG EKF before correction (fit samples)", fit_error)
        print_metrics("GPS − SBG EKF after correction (fit samples)", fit_error - bias)
        corrected_pitch = gps_pitch - bias

    difference = gps_pitch - horizon_at_gps

    print_metrics("GPS − horizon", difference[overlap])

    if corrected_pitch is not None:
        corrected_difference = corrected_pitch - horizon_at_gps
        print_metrics("Corrected GPS − horizon", corrected_difference[overlap])

    if len(ekf) >= 2:
        common = overlap & (times >= ekf[0, 0]) & (times <= ekf[-1, 0])
        if common.any():
            reference = np.interp(times[common], *ekf.T)
            print_metrics("GPS − SBG EKF", gps_pitch[common] - reference)
            if corrected_pitch is not None:
                print_metrics("Corrected GPS − SBG EKF", corrected_pitch[common] - reference)
            print_metrics("Horizon − SBG EKF", horizon_at_gps[common] - reference)
        else:
            print("No GPS timestamps overlap both horizon and SBG EKF pitch.")

    fig, (pitch_ax, error_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6), layout="constrained"
    )
    pitch_ax.plot(horizon[:, 0] - start, horizon[:, 1], label="Horizon")
    pitch_ax.plot(times - start, gps_pitch, ".-", label="GPS")
    if corrected_pitch is not None:
        pitch_ax.plot(times - start, corrected_pitch, ".-", label="GPS (offset corrected)")
    if len(ekf):
        pitch_ax.plot(ekf[:, 0] - start, ekf[:, 1], label="SBG EKF", alpha=0.65)
    pitch_ax.set_ylabel("Pitch (°)")
    pitch_ax.legend()
    error_ax.axhline(0, color="0.5", linewidth=1)
    error_ax.plot(times[overlap] - start, difference[overlap], ".-", label="GPS")
    if corrected_pitch is not None:
        error_ax.plot(times[overlap] - start, corrected_difference[overlap], ".-",
                      label="GPS (offset corrected)")
        error_ax.legend()
    error_ax.set_ylabel("GPS − horizon (°)")
    error_ax.set_xlabel("Elapsed time (s)")
    for axis in (pitch_ax, error_ax):
        axis.grid(alpha=0.3)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", nargs="?", type=Path, default=Path("data/nav_and_forward_cam_seg"))
    parser.add_argument("--max-images", type=int, help="Limit the camera frames processed.")
    parser.add_argument("--gps-sign", type=int, choices=(-1, 1), default=1,
                        help="Use -1 if the GPS antenna baseline has the opposite pitch sign.")
    parser.add_argument("--correct-gps-offset", action="store_true",
                        help="Fit and subtract the mean GPS minus SBG EKF pitch error.")
    parser.add_argument("--output", type=Path,
                        default=Path("figures/gps_pitch_vs_horizon.pdf"))
    parser.add_argument("--no-show", action="store_true", help="Save without opening a window.")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive.")
    if not args.bag.exists():
        parser.error(f"Bag does not exist: {args.bag}")

    try:
        figure = compare_and_plot(
            *read_pitch(args.bag, args.max_images, args.gps_sign),
            correct_gps_offset=args.correct_gps_offset,
        )
    except ValueError as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    print(f"Saved plot to {args.output}")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
