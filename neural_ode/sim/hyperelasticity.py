# %% [markdown]
# # Hyperelasticity
# Author: Jørgen S. Dokken and Garth N. Wells
#
# This section shows how to solve the hyperelasticity problem for deformation of a beam.
#
# We will also show how to create a constant boundary condition for a vector function space.
#
# We start by importing DOLFINx and some additional dependencies.
# Then, we create a slender cantilever consisting of hexahedral elements and create the function space `V` for our unknown.

# %%
from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import numpy as np
import ufl
import json
import datetime
import socket

from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.io import XDMFFile

L = 10.0
domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, 1, 1]], [20, 2, 2], mesh.CellType.hexahedron)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))

# %% [markdown]
# We create two python functions for determining the facets to apply boundary conditions to


# %%
def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], L)


fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

# %% [markdown]
# Next, we create a  marker based on these two functions

# %%
# Concatenate and sort the arrays based on facet indices. Left facets marked with 1, right facets with two
marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

# %% [markdown]
# We then create a function for supplying the boundary condition on the left side, which is fixed.

# %%
u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)

# %% [markdown]
# To apply the boundary condition, we identity the dofs located on the facets marked by the `MeshTag`.

# %%
left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bcs = [fem.dirichletbc(u_bc, left_dofs, V)]

# %% [markdown]
# Next, we define the body force on the reference configuration (`B`), and nominal (first Piola-Kirchhoff) traction (`T`).

# %%
B = fem.Constant(domain, default_scalar_type((0, 0, -9.8)))
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

# %% [markdown]
# Define the test and solution functions on the space $V$

# %%
v = ufl.TestFunction(V)
u = fem.Function(V)
# Previous displacement fields for time-stepping (u_n = u at t_n, u_nn = u at t_{n-1})
u_n = fem.Function(V)
u_nn = fem.Function(V)

# Velocity field
v_field = fem.Function(V)

# %% [markdown]
# Define kinematic quantities used in the problem

# %%
# Spatial dimension
d = len(u)

# Identity tensor
I = ufl.variable(ufl.Identity(d))

# Deformation gradient
F = ufl.variable(I + ufl.grad(u))

# Right Cauchy-Green tensor
C = ufl.variable(F.T * F)

# Invariants of deformation tensors
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F))

# %% [markdown]
# Define the elasticity model via a stored strain energy density function $\psi$, and create the expression for the first Piola-Kirchhoff stress:

# %%
# Elasticity parameters
E = default_scalar_type(1.0e4)
nu = default_scalar_type(0.3)
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
# Stored strain energy density (compressible neo-Hookean model)
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
# Stress
# Hyper-elasticity
P = ufl.diff(psi, F)

# %% [markdown]
# ```{admonition} Comparison to linear elasticity
# To illustrate the difference between linear and hyperelasticity, the following lines can be uncommented to solve the linear elasticity problem.
# ```

# %%
# P = 2.0 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * I

# %% [markdown]
# Define the variational form with traction integral over all facets with value 2. We set the quadrature degree for the integrals to 4.

# %%
metadata = {"quadrature_degree": 4}
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)
# Define form F (we want to find u such that F(u) = 0)
F = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx - ufl.inner(v, T) * ds(2)

# --- Dynamics: add inertial term using central finite-difference for acceleration
# Mass density
rho = fem.Constant(domain, default_scalar_type(1.0))
# Time step (will be overridden in time loop if desired)
dt = 0.1
# Acceleration (finite-difference): a = (u - 2*u_n + u_nn)/dt^2. The inertial virtual work is \int rho * a . v dx
a_fd = (u - u_n * 2 + u_nn) / (dt**2)
F += rho * ufl.inner(a_fd, v) * dx

# %% [markdown]
# As the varitional form is non-linear and written on residual form, we use the non-linear problem class from DOLFINx to set up required structures to use a Newton solver.

# %%
problem = NonlinearProblem(F, u, bcs)

# %% [markdown]
# and then create and customize the Newton solver

# %%
solver = NewtonSolver(domain.comm, problem)

# Set Newton solver options
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"


# %% [markdown]
import os

out_dir = ".data/sim/beam2"
os.makedirs(out_dir, exist_ok=True)

# Write basic mesh info (n_cells might not be available everywhere)
try:
    n_cells = int(domain.topology.index_map(domain.topology.dim).size_global)
except Exception:
    n_cells = None

# We'll populate dynamic metadata (dt, nsteps) after they are defined; write a partial file now
partial_meta = {
    "created": datetime.datetime.utcnow().isoformat() + "Z",
    "host": socket.gethostname(),
    "mesh_dim": int(domain.geometry.dim),
    "n_cells": n_cells,
    "E": float(E),
    "nu": float(nu),
    "rho": 1.0,
    "fields": ["displacement", "velocity", "PK1", "energy_density"],
}
with open(os.path.join(out_dir, "metadata.json"), "w") as fh:
    json.dump(partial_meta, fh, indent=2)

# %% [markdown]
# Finally, we solve the problem over several time steps, updating the z-component of the traction

# %%
log.set_log_level(log.LogLevel.INFO)

# Time integration parameters
dt = 0.01
nsteps = 500

# Initialize history: assume initially at rest
u.x.array[:] = 0
u_n.x.array[:] = u.x.array.copy()
u_nn.x.array[:] = u.x.array.copy()
v_field.x.array[:] = 0

# Time steps (wrapped in XDMF writer for time-series output)
# Give names to functions for the XDMF fields
u.name = "displacement"
v_field.name = "velocity"

xdmf_path = os.path.join(out_dir, "solution.xdmf")
# Prepare lower-degree write spaces
V_write = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
u_write = fem.Function(V_write)
v_write = fem.Function(V_write)
# Scalar space for energy density and tensor space for first Piola-Kirchhoff stress
Vs_scalar = fem.functionspace(domain, ("Lagrange", 1))
V_tensor = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, domain.geometry.dim)))
psi_write = fem.Function(Vs_scalar)
P_write = fem.Function(V_tensor)

with XDMFFile(domain.comm, xdmf_path, "w") as xdmf:
    # Write the mesh once
    xdmf.write_mesh(domain)
    # Name the write functions (XDMF field labels)
    u_write.name = "displacement"
    v_write.name = "velocity"
    psi_write.name = "energy_density"
    P_write.name = "PK1"

    # Write initial state at time 0.0: interpolate high-order functions into low-order write space
    u_write.interpolate(u)
    v_write.interpolate(v_field)
    # Interpolate stress and energy density
    P_expr = fem.Expression(P, V_tensor.element.interpolation_points())
    psi_expr = fem.Expression(psi, Vs_scalar.element.interpolation_points())
    P_write.interpolate(P_expr)
    psi_write.interpolate(psi_expr)
    xdmf.write_function(u_write, 0.0)
    xdmf.write_function(v_write, 0.0)
    xdmf.write_function(P_write, 0.0)
    xdmf.write_function(psi_write, 0.0)

    for n in range(1, nsteps + 1):
        # Solve nonlinear system for displacement at new time
        num_its, converged = solver.solve(u)
        assert converged
        u.x.scatter_forward()

        # Compute acceleration (central difference)
        a_array = (u.x.array - 2 * u_n.x.array + u_nn.x.array) / (dt**2)

        # Update velocity (central difference): v^{n+1/2} ~ (u^{n+1} - u^{n})/dt
        v_field.x.array[:] = (u.x.array - u_n.x.array) / dt

        # Shift histories: u_{n-1} <- u_n, u_n <- u
        u_nn.x.array[:] = u_n.x.array.copy()
        u_n.x.array[:] = u.x.array.copy()

        t = n * dt
        print(f"Time step {n}, time {t:.4f}, Number of iterations {num_its}")

        # Write time-dependent functions to XDMF (time series)
        u_write.interpolate(u)
        v_write.interpolate(v_field)
        # interpolate and write stress and energy density
        P_expr = fem.Expression(P, V_tensor.element.interpolation_points())
        psi_expr = fem.Expression(psi, Vs_scalar.element.interpolation_points())
        P_write.interpolate(P_expr)
        psi_write.interpolate(psi_expr)
        xdmf.write_function(u_write, float(t))
        xdmf.write_function(v_write, float(t))
        xdmf.write_function(P_write, float(t))
        xdmf.write_function(psi_write, float(t))

        # (XDMF time-series already written above)
# Update metadata with runtime parameters
full_meta = dict(partial_meta)
full_meta.update({"dt": float(dt), "nsteps": int(nsteps)})
with open(os.path.join(out_dir, "metadata.json"), "w") as fh:
    json.dump(full_meta, fh, indent=2)
# %% [markdown]
# <img src="./deformation.gif" alt="gif" class="bg-primary mb-1" width="800px">
