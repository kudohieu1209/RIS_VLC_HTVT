import math

import numpy as np

from .utils import as_vector, clip_unit, distance, unit_vector
from .vlc_channel import LED_NORMAL, PD_NORMAL, lambertian_order, optical_concentrator_gain


def ap_to_ris_gain(ap_position, ris_position, config) -> float:
    """Geometric gain from LED AP to the RIS patch on the wall y = 0."""

    ap = as_vector(ap_position)
    ris = as_vector(ris_position)
    ris_normal = as_vector(config.ris_normal)
    d = distance(ap, ris)
    if d <= 0.0:
        return 0.0

    ray_ap_to_ris = unit_vector(ris - ap)
    ray_ris_to_ap = unit_vector(ap - ris)
    cos_phi = clip_unit(float(np.dot(LED_NORMAL, ray_ap_to_ris)))
    cos_incident = clip_unit(float(np.dot(ris_normal, ray_ris_to_ap)))

    if cos_phi <= 0.0 or cos_incident <= 0.0:
        return 0.0

    m = lambertian_order(config.led_half_power_angle_deg)
    return (
        ((m + 1.0) * config.ris_effective_area / (2.0 * math.pi * d**2))
        * (cos_phi**m)
        * cos_incident
    )


def ris_to_pd_gain(ris_position, pd_position, config) -> float:
    """Geometric gain from the RIS patch to an upward-facing PD."""

    ris = as_vector(ris_position)
    pd = as_vector(pd_position)
    ris_normal = as_vector(config.ris_normal)
    d = distance(ris, pd)
    if d <= 0.0:
        return 0.0

    ray_ris_to_pd = unit_vector(pd - ris)
    ray_pd_to_ris = unit_vector(ris - pd)
    cos_departure = clip_unit(float(np.dot(ris_normal, ray_ris_to_pd)))
    cos_psi = clip_unit(float(np.dot(PD_NORMAL, ray_pd_to_ris)))

    if cos_departure <= 0.0 or cos_psi <= 0.0:
        return 0.0

    psi_rad = math.acos(cos_psi)
    concentrator_gain = optical_concentrator_gain(psi_rad, config)
    if concentrator_gain <= 0.0:
        return 0.0

    return (
        (config.pd_area_m2 / (math.pi * d**2))
        * cos_departure
        * config.optical_filter_gain
        * concentrator_gain
        * cos_psi
    )


def ris_channel_gain(ap_position, ris_position, pd_position, config, alignment_gain=None) -> float:
    """Simplified AP-RIS-PD channel gain.

    This is an approximate reflection path model for course-level simulation,
    not a full optical hardware model of a real RIS.
    """

    if alignment_gain is None:
        alignment_gain = config.ris_alignment_gain
    alignment_gain = float(np.clip(alignment_gain, 0.0, 1.0))

    h_ap_ris = ap_to_ris_gain(ap_position, ris_position, config)
    h_ris_pd = ris_to_pd_gain(ris_position, pd_position, config)
    return (
        config.ris_reflection_coefficient
        * h_ap_ris
        * h_ris_pd
        * alignment_gain
    )
