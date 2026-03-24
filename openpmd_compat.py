"""
Compatibility layer replacing removed openpmd-viewer internal APIs.
Uses h5py directly to read openPMD files — stable across all versions.
"""
import h5py
import numpy as np


class FieldInfo:
    """Mimics the axis info object returned by old openpmd-viewer internals."""
    def __init__(self, z, r, axes=None, dz=None, dr=None,
                 zmin=None, zmax=None, rmin=None, rmax=None):
        self.z = z
        self.r = r
        self.axes = axes or ["r", "z"]
        self.dz = dz
        self.dr = dr
        self.zmin = zmin
        self.zmax = zmax
        self.rmin = rmin
        self.rmax = rmax


def opmd_read_params(filename):
    """Read time and basic params from an openPMD file."""
    with h5py.File(filename, "r") as f:
        it_keys = list(f["/data"].keys())
        it_key = it_keys[0]
        base = f["/data/" + it_key]
        time = float(base.attrs["time"])
        dt = float(base.attrs["dt"])
        # Check for openPMD extensions (e.g., ED-PIC)
        extensions = []
        if "openPMDextension" in f.attrs:
            ext_val = f.attrs["openPMDextension"]
            if isinstance(ext_val, (int, np.integer)) and ext_val > 0:
                extensions.append("ED-PIC")
            elif isinstance(ext_val, str) and ext_val:
                extensions.append(ext_val)
        params = {
            "time": time,
            "dt": dt,
            "iteration": int(it_key),
            "extensions": extensions,
        }
    return time, params


def opmd_read_raw(file_handle, species, record_comp, extensions=None):
    """Read particle raw data from an already-opened h5py file handle."""
    it_keys = list(file_handle["/data"].keys())
    it_key = it_keys[0]
    base_path = "/data/" + it_key + "/particles/" + species

    # Map common FBPIC/openPMD field names
    comp_map = {
        "w": "weighting",
        "x": "position/x",
        "y": "position/y",
        "z": "position/z",
        "ux": "momentum/x",
        "uy": "momentum/y",
        "uz": "momentum/z",
    }
    mapped = comp_map.get(record_comp, record_comp)

    parts = mapped.split("/")
    if len(parts) == 2:
        record, comp = parts
        dset = file_handle[base_path + "/" + record + "/" + comp]
    else:
        dset = file_handle[base_path + "/" + mapped]

    data = dset[:]
    if "unitSI" in dset.attrs:
        data = data * dset.attrs["unitSI"]

    # For momentum, convert from SI (kg·m/s) to normalized units (γβ)
    if mapped.startswith("momentum/"):
        from scipy.constants import m_e, c
        data = data / (m_e * c)

    return data


def opmd_read_field_circ(filename, field_path, slice_across=None,
                          slice_relative_position=None, m="all", theta=None):
    """Read a cylindrical (RZ) field from an openPMD file.

    Returns (field_data, info).
    If theta is None: field_data shape (n_modes, Nr, Nz).
    If theta is given: field_data shape (Nr, Nz) reconstructed at that angle.
    """
    with h5py.File(filename, "r") as f:
        it_keys = list(f["/data"].keys())
        it_key = it_keys[0]
        field_group = f["/data/" + it_key + "/meshes/" + field_path]
        data = field_group[:]

        field_name = field_path.split("/")[0]
        mesh_group = f["/data/" + it_key + "/meshes/" + field_name]

        grid_spacing = mesh_group.attrs.get("gridSpacing", [1.0, 1.0])
        grid_offset = mesh_group.attrs.get("gridGlobalOffset", [0.0, 0.0])
        unit_SI = field_group.attrs.get("unitSI", 1.0)

        # openPMD thetaMode: data shape is (nmodes, Nz, Nr)
        n_modes = data.shape[0]
        Nz = data.shape[1]
        Nr = data.shape[2]

        dz = float(grid_spacing[0])
        dr = float(grid_spacing[1])
        zmin = float(grid_offset[0])
        rmin = float(grid_offset[1])

        pos = field_group.attrs.get("position", [0.0, 0.0])
        z = zmin + (np.arange(Nz) + pos[0]) * dz
        r = rmin + (np.arange(Nr) + pos[1]) * dr

        data = data * unit_SI

    info = FieldInfo(
        z=z, r=r, dz=dz, dr=dr,
        zmin=z[0], zmax=z[-1],
        rmin=r[0], rmax=r[-1],
    )

    if theta is not None:
        result = data[0].copy()
        n_azimuthal = (n_modes + 1) // 2
        for im in range(1, n_azimuthal):
            result += data[2 * im - 1] * np.cos(im * theta)
            result += data[2 * im] * np.sin(im * theta)
        return result.T, info
    else:
        return np.transpose(data, (0, 2, 1)), info


def opmd_comb_cyl(Fr, Ft, theta, coord, info):
    """Combine cylindrical Fr, Ft into Cartesian at angle theta."""
    if theta is None:
        theta = 0.0
    if coord == "x":
        return Fr * np.cos(theta) - Ft * np.sin(theta)
    elif coord == "y":
        return Fr * np.sin(theta) + Ft * np.cos(theta)
    else:
        raise ValueError("coord must be 'x' or 'y', got '" + coord + "'")
