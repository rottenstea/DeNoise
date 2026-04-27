import pandas as pd
import numpy as np
from .coordinate_transforms import (
    transform_sphere_to_cartesian,
    transform_gal_cartesian_and_vtan_to_icrs_pm,
    recenter_ra,
    restore_ra,
)


def add_uniform_noise(noise_percentage: float, cluster_df: pd.DataFrame, label_col: str,
                      pos_range_extend_p: float = 0.2, vel_range_extend: int = 5, parameters: list = None,
                      spherical_parameters: bool = True):
    """Append uniformly distributed noise rows to a stellar cluster DataFrame.

    Samples ``int(noise_percentage * len(cluster_df))`` points uniformly inside an
    enlarged bounding box of the cluster: position bounds are the per-axis
    ``[min, max]`` extended by ``pos_range_extend_p`` on either side (parallax is
    clipped at 0), and velocity bounds are ``mean ± vel_range_extend * std``.
    Sampled rows are then transformed so each output row carries both ICRS
    spherical (ra, dec, parallax, pmra, pmdec) and Galactic-Cartesian
    (X, Y, Z, v_a_lsr, v_d_lsr) columns. Noise rows are flagged with ``-1`` in
    ``label_col`` and appended with ``ignore_index=True``.

    Parameters
    ----------
    noise_percentage : float
        Fraction (not percent) of noise relative to cluster size, e.g. ``0.1``
        for 10 %.
    cluster_df : pd.DataFrame
        Cluster catalogue. Must contain the columns named in ``parameters``
    label_col : str
        Name of the label column.
    pos_range_extend_p : float, default 0.2
        Fractional extension of the position bounding box on each side of the
        per-axis min/max.
    vel_range_extend : int, default 5
        Number of standard deviations used for the velocity bounds.
    parameters : list of str, optional
        Five column names ``[pos1, pos2, pos3, vel1, vel2]``. Defaults to
        ``['ra', 'dec', 'parallax', 'pmra', 'pmdec']``.
    spherical_parameters : bool, default True
        ``True`` if ``parameters`` are ICRS spherical, ``False`` if they are
        Galactic Cartesian (X, Y, Z, v_a_lsr, v_d_lsr). Selects the transform
        direction.

    Returns
    -------
    pd.DataFrame
        ``cluster_df`` with the noise rows appended.

    """
    # --- Setup ---
    if parameters is None:
        parameters = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']

    # Step 1: Bounds for position
    pos = cluster_df[parameters[:3]].to_numpy(copy=True)

    # Handle RA wrap-around: recenter on the circular mean so the cluster is
    # contiguous regardless of whether it crosses 0°/360°.
    ra_center = None
    if spherical_parameters and parameters[0].lower() == "ra":
        pos[:, 0], ra_center = recenter_ra(pos[:, 0])

    min_pos = pos.min(axis=0)
    max_pos = pos.max(axis=0)
    range_pos = max_pos - min_pos
    low_pos = min_pos - pos_range_extend_p * range_pos
    high_xyz = max_pos + pos_range_extend_p * range_pos

    if parameters[2].lower() in ["parallax", "plx"]:
        low_pos[2] = max(0, low_pos[2])  # do not allow negative parallaxes

    # Step 2: Bounds for velocity1, velocity2
    vel = cluster_df[parameters[3:]].values
    mean_vel = vel.mean(axis=0)
    std_vel = vel.std(axis=0)
    low_vel = mean_vel - vel_range_extend * std_vel
    high_vel = mean_vel + vel_range_extend * std_vel

    # Step 3: Generate noise points
    n_noise = int(noise_percentage * len(cluster_df))  # e.g., 10% noise per cluster
    noise_pos = np.random.uniform(low=low_pos, high=high_xyz, size=(n_noise, 3))
    if ra_center is not None:
        noise_pos[:, 0] = restore_ra(noise_pos[:, 0], ra_center)
    noise_vel = np.random.uniform(low=low_vel, high=high_vel, size=(n_noise, 2))
    noise_data = np.hstack([noise_pos, noise_vel])

    # Step 4: If input data is spherical, calculate the galactic cartesian values
    if spherical_parameters:
        spherical_df = pd.DataFrame(noise_data, columns=parameters)

        cartesian_df = transform_sphere_to_cartesian(ra=noise_data[:, 0],
                                                     dec=noise_data[:, 1],
                                                     parallax=noise_data[:, 2],
                                                     pmra=noise_data[:, 3],
                                                     pmdec=noise_data[:, 4], )
    # Else, calculate the ICRS values
    else:
        cartesian_df = pd.DataFrame(noise_data, columns=parameters)

        spherical_df = transform_gal_cartesian_and_vtan_to_icrs_pm(X=noise_data[:, 0],
                                                                   Y=noise_data[:, 1],
                                                                   Z=noise_data[:, 2],
                                                                   v_a_lsr=noise_data[:, 3],
                                                                   v_d_lsr=noise_data[:, 4], )

    # Step 5: Combine spherical and Cartesian
    df_noise = pd.concat([spherical_df.reset_index(drop=True),
                          cartesian_df.reset_index(drop=True)], axis=1)

    # Step 6: Add label column for noise
    df_noise[label_col] = -1

    # Append to original DataFrame
    noisy_cluster_df = pd.concat([cluster_df, df_noise], ignore_index=True)

    return noisy_cluster_df
