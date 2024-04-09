# fea_fundamentals


## Files for Bower's sample code

(Refer to https://solidmechanics.org/FEA.php for more)

1. **FEM_1D_Static.m** Simple static analysis of a 1D bar subjected to axial body force
2. **FEM_1D_newmark.m** Simple dynamic analysis of a 1D bar subjected to axial body force, using Newmark time integration.
3. **FEM_1D_modal.m**  Simple dynamic analysis of a 1D bar subjected to axial body force, using modal time integration.
4. **FEM_conststrain.m**   Simple 2D FEA code with constant strain triangles.  Should be run with the input file **FEM_conststrain.txt** or (for a larger problem) **FEM_conststrain_holeplate.txt**
5. The following files all solve 2D or 3D static linear elastic problems, but illustrate various refinements of the finite element method:
    1. **FEM_2Dor3D_linelast_standard.m** 2D(plane strain/stress) or 3D static linear elasticity code with fully integrated elements. The code can be run with the following input files.

        - **Linear_elastic_triangles.txt**: 2D plane strain problem with two triangular elements;
        - **Linear_elastic_quad4.txt**: 2D plane strain problem with 4 noded quadrilateral elements;
        - **Linear_elastic_quad8.txt**: 2D plane strain problem with 8 noded quadrilateral elements;
        - **Linear_elastic_brick8.txt**: 3D problem with 8 noded brick elements;
        - **Linear_elastic_brick20.txt**: 3D problem with 20 noded brick elements;
        - **Linear_elastic_pressurized_cylinder.txt**: 2D simulation of a pressurized cylinder;
    2. **FEM_shear_locking_demo.m** - Solves the beam bending problem discussed in Section 8.6.2, and compares the FEM solution with the exact solution to illustrate shear locking. This version of the code must be run with **shear_locking_demo_linear.txt** (solution with 4 noded quad elements).

    3. **FEM_incompatible_modes.m** - Solves the beam bending problem discussed in Section 8.6.2 using incompatible mode elements, and compares the FEM solution with the exact solution to demonstrate that the elements avoid shear locking. This version of the code must be run with **shear_locking_demo_linear.txt** (solution with 4 noded quad elements).

    4. **FEM_volumetric_locking_demo.m** - Solves the pressurized cylindrical cavity problem discussed in Section 8.6.2, and compares the FEM solution with the exact solution. This version of the code must be run with **volumetric_locking_demo_linear.txt** (solution with 4 noded quad elements) or **volumetric_locking_demo_quadratic.txt** (solution with 8 noded quadrilateral elements).

    5. **FEM_hourglassing_demo.m** - Solves the pressurized cylindrical cavity problem discussed in Section 8.6.2 with reduced integration elements, demonstrating hourglassing. This version of the code must be run with **volumetric_locking_demo_linear.txt** (solution with 4 noded quad elements).

    6. **FEM_selective_reduced_integration.m** - Solves the pressurized cylindrical cavity problem discussed in Section 8.6.2 using selectively reduced integration, and compares the FEM solution with the exact solution. This version of the code must be run with **volumetric_locking_demo_quadratic.txt** (solution with 8 noded quadrilateral elements).

    7. **FEM_hourglasscontrol.m** - illustrates use of hourglass control to eliminate hourglassing in 4 noded quadrilateral elements.  This version of the code must be run with **volumetric_locking_demo_linear.txt** (solution with 4 noded quad elements)

    8. **FEM_Bbar.m** – Solves the pressurized cylinder problem discussed in Section 8.6.2 using the B-bar method, and compares the solution with the exact solution. This version of the code must be run with **volumetric_locking_demo_linear.txt** or **volumetric_locking_demo_quadratic.txt**.

    9. **FEM_hybrid.m** – Solves the pressurized cylinder problem discussed in Section 8.6.2 using hybrid elements, and compares the FEM solution with the exact solution. This version of the code must be run with **volumetric_locking_demo_linear.txt** or **volumetric_locking_demo_quadratic.txt**.

6. **FEM_2Dor3D_linelast_dynamic.m**: Solves 2D or 3D dynamic linear elasticity problems, using Newmark time integration. The code can be run with the input file **Linear_elastic_dynamic_beam.txt**.

7. **FEM_2Dor3D_modeshapes.m**: Calculates mode shapes and natural frequencies for a linear elastic solid. The code can be run with the input file **Linear_elastic_dynamic_beam.txt**.

8. **FEM_2Dor3D_hypoelastic_static.m**: Solves 2D (plane strain only) or 3D static problems for a hypoelastic material, as discussed in Section 8.3.9. The input file is **Hypoelastic_quad4.txt**.

9. **FEM_2Dor3D_hyperelastic_static.m**: Solves 2D (plane strain only) or 3D static problems for a hyperelastic (Neo-Hookean) material. An input file is provided in **Hyperelastic_quad4.txt**.

10. **FEM_2Dor3D_viscoplastic_static.m**: Solves 2D (plane strain only) or 3D static problems for a small strain viscoplastic material. An input file is provided in **Viscoplastic_quad4.txt**