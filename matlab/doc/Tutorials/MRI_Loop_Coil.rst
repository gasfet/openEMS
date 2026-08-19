.. _octave_tutorial_mri_loop_coil:

MRI Loop Coil
=============

Simulate a single-loop MRI receive coil at 297 MHz (7 T), compute the
B1 field distribution, and evaluate the coil's sensitivity and coupling.

**This tutorial covers:**

* Circular loop coil geometry with lumped capacitors for resonance tuning
* Lumped port excitation and S11 minimization
* B1 field dump and sensitivity mapping in the coil plane

Octave/Matlab Script
--------------------

.. include:: ./__MRI_Loop_Coil.txt

Images
------

.. figure:: images/MRI_Loop_Coil_Setup.png
    :width: 80%
    :alt: Loop coil setup geometry

    Loop coil geometry with lumped capacitors (AppCSXCAD)

.. figure:: images/MRI_Loop_Coil_B1_xy.png
    :width: 80%
    :alt: B1 field in xy-plane

    B1 field distribution in the xy-plane (coil plane)

.. figure:: images/MRI_Loop_Coil_B1_xz.png
    :width: 80%
    :alt: B1 field in xz-plane

    B1 field distribution in the xz-plane

.. figure:: images/MRI_Loop_Coil_SAR.png
    :width: 80%
    :alt: SAR distribution

    SAR distribution in the phantom

.. figure:: images/MRI_Loop_Coil_3D_SAR.png
    :width: 80%
    :alt: 3D SAR distribution

    3D SAR distribution

Body Model
----------

This tutorial requires the **Ella** voxel body model from the IT'IS Virtual
Family dataset (``Ella_26y_V2_1mm``). The dataset is free for academic and
non-commercial use but requires registration and a license agreement from the
IT'IS Foundation (https://itis.swiss/virtual-population/).

Once downloaded, the raw files are converted to an openEMS HDF5 DiscMaterial
file by ``Convert_VF_DiscMaterial`` (run once, result is cached).

**Automatic phantom fallback:**

If the VF dataset is not installed, the script falls back automatically to
the bundled ``resources/phantoms/phantom_head_298MHz.h5`` — a 3-layer
ellipsoidal head phantom (skin / skull / brain) with tissue properties at
298 MHz from the IT'IS database, shared with the Python interface. The
phantom uses the same HDF5 DiscMaterial format as the converted VF model,
so the full B1 and SAR workflow runs without any changes.

Literature
----------

* A. Christ et al., "The Virtual Family — Development of surface-based
  anatomical models of two adults and two children for dosimetric simulations,"
  *Phys. Med. Biol.*, vol. 55, 2010.
