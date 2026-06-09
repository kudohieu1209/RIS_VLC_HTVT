import math

import numpy as np

from .utils import as_vector, clip_unit, distance, unit_vector


LED_NORMAL = np.array([0.0, 0.0, -1.0])
PD_NORMAL = np.array([0.0, 0.0, 1.0])


def lambertian_order(half_power_angle_deg: float) -> float:
    half_power_rad = math.radians(half_power_angle_deg)
    cos_half = math.cos(half_power_rad)
    if cos_half <= 0.0 or math.isclose(cos_half, 1.0):
        raise ValueError("Half-power angle must be between 0 and 90 degrees.")
    return -math.log(2.0) / math.log(cos_half)


def optical_concentrator_gain(psi_rad: float, config) -> float:
    fov_rad = math.radians(config.pd_fov_deg)
    if psi_rad < 0.0 or psi_rad > fov_rad:
        return 0.0
    sin_fov = math.sin(fov_rad)
    if sin_fov <= 0.0:
        return 0.0
    return (config.optical_concentrator_index**2) / (sin_fov**2)


def los_channel_gain(ap_position, pd_position, config) -> float:
    """Lambertian LoS channel gain for a downward LED and upward PD."""

    ap = as_vector(ap_position)
    pd = as_vector(pd_position)
    separation = pd - ap
    d = distance(ap, pd)
    if d <= 0.0:
        return 0.0

    ray_ap_to_pd = unit_vector(separation)
    ray_pd_to_ap = unit_vector(-separation)
    cos_phi = clip_unit(float(np.dot(LED_NORMAL, ray_ap_to_pd)))
    cos_psi = clip_unit(float(np.dot(PD_NORMAL, ray_pd_to_ap)))

    if cos_phi <= 0.0 or cos_psi <= 0.0:
        return 0.0

    psi_rad = math.acos(cos_psi)
    concentrator_gain = optical_concentrator_gain(psi_rad, config)
    if concentrator_gain <= 0.0:
        return 0.0

    m = lambertian_order(config.led_half_power_angle_deg)
    return (
        ((m + 1.0) * config.pd_area_m2 / (2.0 * math.pi * d**2))
        * (cos_phi**m)
        * config.optical_filter_gain
        * concentrator_gain
        * cos_psi
    )
