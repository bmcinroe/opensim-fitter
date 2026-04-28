import numpy as np
import casadi as ca
import opensim as osim
import matplotlib.pyplot as plt

# Load the IK solution and extract the first column.
table = osim.TimeSeriesTable('jump_1_ik_solution.sto')
times = table.getIndependentColumn()
col1 = table.getDependentColumnAtIndex(0).to_numpy()
col2 = table.getDependentColumnAtIndex(2).to_numpy()
n = table.getNumRows()
dt = times[1] - times[0]

class MyQFunc(ca.Callback):
    """Maps joint angles q -> observations y at a single time step.

    Extend get_sparsity_in/out, eval, and get_jacobian for multi-DOF FK:
      - get_sparsity_in:  ca.Sparsity.dense(n_dof, 1)
      - get_sparsity_out: ca.Sparsity.dense(n_obs, 1)
      - eval: call OpenSim FK with np.array(arg[0]).flatten()
      - get_jacobian: return d(FK)/dq as a ca.Function
    """
    def __init__(self):
        ca.Callback.__init__(self)
        self.construct("MyQFunc", {})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):  return ca.Sparsity.dense(1, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)

    def eval(self, arg):
        q = float(arg[0])
        y = q          # replace with FK evaluation
        return [ca.DM(y)]

    def has_jacobian(self): return True

    def get_jacobian(self, name, inames, onames, opts):
        # Inputs: primal input q and primal output y (available if J depends on them).
        # Output: J = dy/dq, shape (n_obs, n_dof).
        q   = ca.MX.sym("q",   1, 1)
        out = ca.MX.sym("out", 1, 1)
        J = ca.DM([[1.0]])   # d(q)/dq; replace with analytic dFK/dq
        return ca.Function(name, [q, out], [J], inames, onames)

my_q_func = MyQFunc()

# Fitting parameters.
t_data = np.linspace(0, 1, n)
degree = 3
knot_interval = 0.05
nc = int(times[-1] / knot_interval)

# Clamped knot vector. For n control points and degree p, there are n+p+1 knots.
# The first and last p+1 knots are clamped to 0 and 1, respectively, and the interior
# knots are uniformly spaced in (0, 1).
knots = np.concatenate([
    np.repeat(0.0, degree),
    np.linspace(0, 1, nc - degree + 1),
    np.repeat(1.0, degree),
])

# Build basis matrix B[i,j] = N_j(t_i) numerically.
# ca.bspline does not support symbolic differentiation w.r.t. coefficients, so
# we evaluate each basis function by passing a unit coefficient vector.
# Symbolic spline — used both to build B and for evaluation after fitting.
t_sym = ca.MX.sym("t")
c = ca.MX.sym("c", nc, 2)

# Scalar spline function for building B matrix.
c_scalar = ca.MX.sym("c_scalar", nc, 1)
spline_scalar_expr = ca.bspline(t_sym, c_scalar, [knots], [degree], 1)
spline_fn = ca.Function("spline", [t_sym, c_scalar], [spline_scalar_expr])

# Build basis matrix B[i,j] = N_j(t_i) by evaluating with unit coefficient vectors.
B = np.zeros((n, nc))
for j in range(nc):
    e_j = np.zeros(nc); e_j[j] = 1.0
    B[:, j] = [float(spline_fn(ti, e_j)) for ti in t_data]

# Optimization: min_c ||B @ c - y||^2 for both columns simultaneously.
B_dm = ca.DM(B)
q_pred = B_dm @ c   # n x 2

col_fns = [my_q_func.map(n)(q_pred[:, k].T).T for k in range(2)]
y_pred = ca.horzcat(*col_fns)   # n x 2

y_data = ca.DM(np.column_stack([col1, col2]))
cost = ca.sumsqr(y_pred - y_data)

solver = ca.nlpsol("solver", "ipopt", {"x": ca.vec(c), "f": cost},
                   {"ipopt.print_level": 7, "ipopt.sb": "yes"})
sol = solver(x0=np.zeros(nc * 2))
ctrl_pts = np.array(sol["x"]).reshape(2, nc).T

# Evaluate the fitted splines on a dense grid.
t_dense = np.linspace(0, 1, 500)
y_fit = np.array([[float(spline_fn(ti, ctrl_pts[:, k])) for ti in t_dense] for k in range(2)])

# Plot
col_labels = [table.getColumnLabel(0), table.getColumnLabel(1)]
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for k, ax in enumerate(axes):
    data_col = [col1, col2][k]
    ax.plot(t_data, data_col, 'o', markersize=3, label='Original data', alpha=0.6)
    ax.plot(t_dense, y_fit[k], '-', linewidth=2, label=f'B-spline fit (deg={degree}, nc={nc})')
    ax.set_ylabel(col_labels[k])
    ax.legend()
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Normalized time')
plt.tight_layout()
plt.show()
