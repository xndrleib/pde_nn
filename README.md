## Problem Formulation
**Heat Distribution in a Thin Metal Plate**

The generalized Poisson equation for heat distribution, considering a non-constant thermal conductivity $k(x, y)$, is given by:
\[
\nabla \cdot (k(x, y) \nabla u(x, y)) = -q(x, y),
\]
where:
- $\nabla \cdot (k(x, y) \nabla u(x, y))$ represents the divergence of the heat flux.
- $q(x, y)$ is the heat source distribution within the plate.
- $x$ and $y$ are the spatial coordinates.
- $u(x, y)$ is the temperature distribution.

In many practical scenarios, especially with metal plates, certain simplifications can be made. Most metals used in engineering applications are both isotropic and homogeneous. This means that:
- **Isotropic Thermal Conductivity**: The thermal conductivity $k$ does not depend on the direction of heat flow within the material, simplifying the modeling process.
- **Homogeneous Material**: The material properties, including thermal conductivity, are uniform throughout the plate, meaning $k(x, y) = k$ is constant across the domain.

Given these properties, the thermal conductivity can be reasonably approximated as a constant. This leads to a simplified version of the Poisson equation:
\[
\Delta u(x, y) = -\frac{q(x, y)}{k}
\]
where $\Delta u(x, y)$ is the Laplacian of the temperature function, and $k$ is now treated as a constant value.

## Task

Solve a 2D Poisson equation to model heat distribution in a thin plate using Neural Networks. 

The problem requires capturing spatially structured features, and the solution should be smooth and continuous, reflecting the physical properties of heat distribution.

## Data

Data that was used for training and validation can be found at [Google Drive](https://drive.google.com/drive/folders/1-d3pXD5OPcvfZacoEO1Y4nfTqnbxwSK7?usp=sharing)

## References
- This repo is based on the code kingly provided by CERFACS in their [PlasmaNet repository](https://gitlab.com/cerfacs/plasmanet).
