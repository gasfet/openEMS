#!/usr/bin/env python3
"""
Create simple layered phantom HDF5 files for use with openEMS AddDiscMaterial
in the MRI tutorials, as a fallback when the IT'IS Virtual Family dataset is
not available.

Generates:
  phantom_head_298MHz.h5  — 3-layer ellipsoidal head phantom at 298 MHz (7 T)
  phantom_body_128MHz.h5  — 3-layer cylindrical body phantom at 128 MHz (3 T)

HDF5 layout (CSXCAD DiscMaterial version 2):
  /               attr Version = 2.0  (float64)
  /DiscData       uint8 array, shape (nz, ny, nx) in C/numpy order
                  value = tissue index (0 = background/air)
    attr DB_Size  int32: number of tissues
    attr epsR     float32[DB_Size]: relative permittivity per tissue
    attr kappa    float32[DB_Size]: conductivity (S/m) per tissue
    attr density  float32[DB_Size]: mass density (kg/m^3) per tissue
    attr Name     str: comma-separated tissue names
  /mesh/x         float32[nx+1]: x cell-boundary positions in metres
  /mesh/y         float32[ny+1]: y cell-boundary positions in metres
  /mesh/z         float32[nz+1]: z cell-boundary positions in metres

Tissue properties are taken from the IT'IS tissue database (Gabriel model).

Usage:
  python3 create_phantoms.py
"""

import os
import numpy as np
import h5py


def write_disc_material(filename, data_xyz, mesh_x, mesh_y, mesh_z,
                        db_epsR, db_kappa, db_density, db_names):
    """Write a CSXCAD DiscMaterial HDF5 file (version 2).

    data_xyz  : uint8 ndarray, shape (nx, ny, nz), data_xyz[ix,iy,iz] = tissue index
    mesh_x/y/z: 1-D float64 arrays of cell-boundary positions in metres,
                lengths nx+1, ny+1, nz+1 respectively
    db_*      : per-tissue property lists/arrays, index 0 = background
    """
    nx, ny, nz = data_xyz.shape
    assert mesh_x.size == nx + 1, "mesh_x must have nx+1 elements"
    assert mesh_y.size == ny + 1, "mesh_y must have ny+1 elements"
    assert mesh_z.size == nz + 1, "mesh_z must have nz+1 elements"

    # CSXCAD C++ reads the flat array with x varying fastest:
    #   flat_index = ix + iy*nx + iz*nx*ny
    # h5py writes numpy C-order (last axis fastest), so we need shape (nz,ny,nx)
    # so that data_c[iz,iy,ix] is at flat C position iz*ny*nx + iy*nx + ix.
    data_c = np.ascontiguousarray(data_xyz.transpose(2, 1, 0), dtype=np.uint8)
    assert data_c.shape == (nz, ny, nx)

    if os.path.exists(filename):
        os.remove(filename)

    with h5py.File(filename, 'w') as f:
        f.attrs['Version'] = np.float64(2.0)

        ds = f.create_dataset('/DiscData', data=data_c,
                              dtype=np.uint8,
                              compression='gzip', compression_opts=9)
        ds.attrs['DB_Size']  = np.int32(len(db_epsR))
        ds.attrs['epsR']     = np.array(db_epsR,    dtype=np.float32)
        ds.attrs['kappa']    = np.array(db_kappa,   dtype=np.float32)
        ds.attrs['density']  = np.array(db_density, dtype=np.float32)
        ds.attrs['Name']     = ','.join(db_names)

        f.create_dataset('/mesh/x', data=mesh_x.astype(np.float32))
        f.create_dataset('/mesh/y', data=mesh_y.astype(np.float32))
        f.create_dataset('/mesh/z', data=mesh_z.astype(np.float32))

    sz = os.path.getsize(filename)
    print(f'  {filename}  ({nx}x{ny}x{nz} voxels, {sz//1024} kB on disk)')


def make_head_phantom(filename):
    """3-layer ellipsoidal head phantom for the 7 T loop-coil tutorial (298 MHz).

    Tissue indices
    --------------
    0  Background (air)
    1  Brain
    2  Skull (cortical bone)
    3  Skin

    Tissue properties at 298 MHz from the IT'IS database (Gabriel model):
      Brain (grey+white avg): eps_r = 52.7, sigma = 0.95 S/m, rho = 1046 kg/m^3
      Skull (cortical):       eps_r = 11.4, sigma = 0.29 S/m, rho = 1908 kg/m^3
      Skin:                   eps_r = 38.9, sigma = 0.84 S/m, rho = 1100 kg/m^3

    Geometry (all semi-axes in metres, phantom centred at origin):
      Outer ellipsoid (skin boundary):      [0.100, 0.080, 0.100]
      Skull/skin boundary:                  [0.091, 0.071, 0.091]
      Brain/skull boundary:                 [0.082, 0.062, 0.082]
    """
    # 2.5 mm voxel resolution, domain covers head + margin
    res = 2.5e-3
    nx_cells, ny_cells, nz_cells = 92, 72, 92   # 230 x 180 x 230 mm
    mesh_x = np.linspace(-0.115, 0.115, nx_cells + 1)
    mesh_y = np.linspace(-0.090, 0.090, ny_cells + 1)
    mesh_z = np.linspace(-0.115, 0.115, nz_cells + 1)

    xc = 0.5 * (mesh_x[:-1] + mesh_x[1:])
    yc = 0.5 * (mesh_y[:-1] + mesh_y[1:])
    zc = 0.5 * (mesh_z[:-1] + mesh_z[1:])

    Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')

    # Normalised ellipsoidal radius for each shell boundary
    a_s, b_s, c_s = 0.100, 0.080, 0.100   # skin outer
    a_k, b_k, c_k = 0.091, 0.071, 0.091   # skull outer (skin inner)
    a_b, b_b, c_b = 0.082, 0.062, 0.082   # brain outer (skull inner)

    r_skin  = np.sqrt((Xc/a_s)**2 + (Yc/b_s)**2 + (Zc/c_s)**2)
    r_skull = np.sqrt((Xc/a_k)**2 + (Yc/b_k)**2 + (Zc/c_k)**2)
    r_brain = np.sqrt((Xc/a_b)**2 + (Yc/b_b)**2 + (Zc/c_b)**2)

    data = np.zeros((nx_cells, ny_cells, nz_cells), dtype=np.uint8)
    data[r_skin  <= 1.0] = 3   # skin
    data[r_skull <= 1.0] = 2   # skull
    data[r_brain <= 1.0] = 1   # brain

    write_disc_material(
        filename, data, mesh_x, mesh_y, mesh_z,
        db_epsR    = [1.0,  52.7, 11.4, 38.9],
        db_kappa   = [0.0,  0.95, 0.29, 0.84],
        db_density = [0.0,  1046, 1908, 1100],
        db_names   = ['Background', 'Brain', 'Skull', 'Skin'],
    )


def make_body_cylinder(filename):
    """3-layer cylindrical body phantom for the 3 T birdcage tutorial (128 MHz).

    Tissue indices
    --------------
    0  Background (air)
    1  Tissue (average soft tissue / brain)
    2  Bone (cortical)
    3  Skin / outer muscle

    Tissue properties at 128 MHz from the IT'IS database (Gabriel model):
      Tissue (soft avg):    eps_r = 62.8, sigma = 0.66 S/m, rho = 1046 kg/m^3
      Bone (cortical):      eps_r = 12.5, sigma = 0.13 S/m, rho = 1908 kg/m^3
      Skin:                 eps_r = 46.8, sigma = 0.58 S/m, rho = 1100 kg/m^3

    Geometry (radii in metres, cylinder centred at origin, axis = z):
      Outer cylinder (skin boundary):  r = 0.100 m
      Bone/skin boundary:              r = 0.092 m
      Tissue/bone boundary:            r = 0.082 m
      z extent: -0.130 to +0.130 m  (260 mm, covers BC.length/2 = 125 mm)
    """
    nx_cells, ny_cells, nz_cells = 92, 92, 104   # 230 x 230 x 260 mm at 2.5 mm
    mesh_x = np.linspace(-0.115, 0.115, nx_cells + 1)
    mesh_y = np.linspace(-0.115, 0.115, ny_cells + 1)
    mesh_z = np.linspace(-0.130, 0.130, nz_cells + 1)

    xc = 0.5 * (mesh_x[:-1] + mesh_x[1:])
    yc = 0.5 * (mesh_y[:-1] + mesh_y[1:])
    zc = 0.5 * (mesh_z[:-1] + mesh_z[1:])

    Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')
    R = np.sqrt(Xc**2 + Yc**2)

    r_outer  = 0.100
    r_bone   = 0.092
    r_tissue = 0.082

    data = np.zeros((nx_cells, ny_cells, nz_cells), dtype=np.uint8)
    data[R <= r_outer]  = 3   # skin/outer muscle
    data[R <= r_bone]   = 2   # cortical bone
    data[R <= r_tissue] = 1   # soft tissue core

    write_disc_material(
        filename, data, mesh_x, mesh_y, mesh_z,
        db_epsR    = [1.0,  62.8, 12.5, 46.8],
        db_kappa   = [0.0,  0.66, 0.13, 0.58],
        db_density = [0.0,  1046, 1908, 1100],
        db_names   = ['Background', 'Tissue', 'Bone', 'Skin'],
    )


if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print('Creating MRI tutorial phantoms ...')
    make_head_phantom(os.path.join(out_dir, 'phantom_head_298MHz.h5'))
    make_body_cylinder(os.path.join(out_dir, 'phantom_body_128MHz.h5'))
    print('Done.')
