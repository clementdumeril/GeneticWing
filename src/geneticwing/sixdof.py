"""Six-degree-of-freedom rigid-body sailplane, with the aerodynamics of PX4's Advanced Plane.

The point mass of :mod:`geneticwing.simulation` assumes the autopilot reaches
the commanded lift coefficient and bank at once (or after a 0.3 s lag). This
module flies the same commands through a rigid body: attitude, body rates,
control surfaces and servos, so that a plan made with the point mass can be
flown by an airframe that has to roll, pitch and yaw to follow it.

State (12, plus 3 servo positions)
----------------------------------
``x = [p_n, p_e, p_d, u, v, w, phi, theta, psi, p, q, r]``: NED position,
body-frame velocity (FRD: x forward, y right wing, z down), Euler angles
(Z-Y-X), body rates. Inputs ``[d_e, d_a, d_r, T]``: elevator, aileron
(differential, positive rolls right), rudder (positive yaws right), thrust
along the body x axis.

Equations (Beard and McLain, *Small Unmanned Aircraft*, ch. 3-4)::

    p_dot  = R(phi, theta, psi) [u v w]
    v_dot  = [r v - q w, p w - r u, q u - p v] + F / m       (F: aero + thrust + gravity, body frame)
    phi_dot   = p + (q sin phi + r cos phi) tan theta
    theta_dot = q cos phi - r sin phi
    psi_dot   = (q sin phi + r cos phi) / cos theta
    J w_dot = M - w x J w                                     (J with Jxz, Beard's Gamma terms)

Aerodynamics: a line-by-line port of gz-sim's ``AdvancedLiftDrag`` system
(the plugin of PX4's ``gz_advanced_plane``, gz-sim8
``src/systems/advanced_lift_drag/AdvancedLiftDrag.cc``): angle of attack and
sideslip from the air velocity at the centre of gravity, the linear
coefficient build-up below stall blended by a sigmoid into a flat-plate model
above it, body-rate and control derivatives, forces in the stability frame,
moments in the body frame. Three deliberate differences, all documented on
:class:`SixDofAirframe`:

* derivatives are per radian of deflection (the plugin's are per degree);
* each surface pair is one input (the plugin sums left and right ailerons,
  whose linear drag terms cancel), and a quadratic control drag
  ``CD_ctrl2 * delta^2`` is added so that deflected surfaces cost energy;
* the stall blending sharpness ``M`` is a parameter: the plugin's default
  (15 per rad) already blends 20 % of the flat-plate model at cruise for a
  wing that stalls at 9 deg, which would cut a sailplane's lift.

The wind (thermals included) enters through the air velocity, as in the
plugin, which only knows a uniform world wind; here any
:class:`~geneticwing.wind.base.WindField` (ENU) is sampled at the centre of
gravity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from geneticwing.aircraft import GRAVITY, SBXC_APPROX, Aircraft

# State indices
PN, PE, PD, U, V, W, PHI, THETA, PSI, P, Q, R = range(12)
DE, DA, DR = 12, 13, 14  # servo positions in the simulation state
STATE_NAMES = ("p_n", "p_e", "p_d", "u", "v", "w", "phi", "theta", "psi", "p", "q", "r")
INPUT_NAMES = ("d_e", "d_a", "d_r", "T")


@dataclass(frozen=True)
class SixDofAirframe:
    """Mass, inertia, geometry and aerodynamic derivatives (stability axes, per radian).

    Coefficient names follow the PX4 plugin: ``C`` + force or moment (``L, D, Y``
    or ``ell, em, en``) + variable (``a`` alpha, ``b`` beta, ``p q r`` body
    rates non-dimensionalized by ``b / 2V`` or ``c / 2V``, ``de da dr``
    surfaces).
    """

    name: str
    mass: float  # kg
    jx: float  # kg m^2
    jy: float
    jz: float
    jxz: float
    area: float  # m^2
    span: float  # m
    mac: float  # m
    oswald: float
    # longitudinal
    CL0: float
    CLa: float
    CLq: float
    CLde: float
    CD0: float
    CDq: float
    Cem0: float
    Cema: float
    Cemq: float
    Cemde: float
    # lateral-directional
    CYb: float
    CYp: float
    CYr: float
    CYdr: float
    Cellb: float
    Cellp: float
    Cellr: float
    Cellda: float
    Celldr: float
    Cenb: float
    Cenp: float
    Cenr: float
    Cenda: float
    Cendr: float
    # quadratic control drag, per rad^2 (not in the plugin, see the module docstring)
    CDde2: float = 0.0
    CDda2: float = 0.0
    CDdr2: float = 0.0
    # stall model (plugin defaults except M)
    alpha_stall: float = 0.15  # rad
    M: float = 50.0  # 1/rad, sigmoid sharpness
    Cema_stall: float | None = None  # pitch slope beyond stall; None = Cema
    CD_fp_k1: float = -0.224
    CD_fp_k2: float = -0.115
    rho: float = 1.225
    g: float = GRAVITY
    # actuators
    deflection_max: tuple[float, float, float] = (np.radians(20.0), np.radians(20.0), np.radians(30.0))
    servo_time_constant: float = 0.05  # s
    servo_rate_max: float = np.radians(300.0)  # rad/s
    thrust_max: float = 15.0  # N
    notes: dict = field(default_factory=dict, compare=False)

    def static_cl(self, alpha):
        """The plugin's blended lift curve (no rates, no surfaces)."""
        alpha = np.asarray(alpha, dtype=float)
        sigma = np.vectorize(lambda a: stall_blend(self, a))(alpha)
        return (1 - sigma) * (self.CL0 + self.CLa * alpha) + sigma * 2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha)

    @property
    def alpha_cl_max(self) -> float:
        """Angle of attack of the blended curve's maximum lift (the stall the autopilot protects)."""
        a = np.linspace(0.0, self.alpha_stall + 0.2, 2001)
        return float(a[np.argmax(self.static_cl(a))])

    @property
    def aspect_ratio(self) -> float:
        return self.span**2 / self.area

    @property
    def weight(self) -> float:
        return self.mass * self.g

    def point_mass(self) -> Aircraft:
        """The point-mass aircraft with the same mass, area and polar."""
        return Aircraft(name=self.name + " (point mass)", mass=self.mass, wing_area=self.area, cd0=self.CD0,
                        k=1.0 / (np.pi * self.aspect_ratio * self.oswald), cl_max=SBXC_APPROX.cl_max,
                        rho=self.rho, g=self.g)


# -- SB-XC estimate --------------------------------------------------------------------------
def sbxc_estimated(static_margin: float = 0.10, dihedral_deg: float = 4.0, inertia_scale: float = 1.0) -> SixDofAirframe:
    """SB-XC derivatives from textbook formulas (Nelson, *Flight Stability and Automatic Control*, ch. 2-5).

    Known: mass 4.22 kg, wing area 1.068 m^2, aspect ratio 19.8 (Model
    Aviation review, 1992); the polar ``C_D0 = 0.018, e = 0.85`` of
    :data:`geneticwing.aircraft.SBXC_APPROX`, kept identical so that a trimmed
    glide sinks exactly as the point mass does. ESTIMATED (no drawing at
    hand): taper 0.5, ailerons on the outer 40 % of each semispan
    (differential, adverse yaw 8 % of C_L times the roll power), a 1.10 m
    tail arm, tail volumes 0.40 (horizontal) and 0.030 (vertical), surfaces
    at about half of their tail's chord, the dihedral, the static margin, and the
    inertia (wing 1.6 kg spread along the span, tail 0.25 kg at the tail
    arm, the rest near the centre of gravity). To replace by AVL runs on the
    real geometry and a swing test.
    """
    ac = SBXC_APPROX
    S, AR, e = ac.wing_area, 19.8, 0.85
    b = math.sqrt(AR * S)
    taper = 0.5
    c_root = 2.0 * S / (b * (1.0 + taper))
    mac = (2.0 / 3.0) * c_root * (1 + taper + taper**2) / (1 + taper)
    l_t = 1.10  # m, tail arm (horizontal and vertical)
    vh, vv = 0.40, 0.030  # tail volume coefficients (a large fin, as RC sailplanes have)
    s_h, s_v = vh * S * mac / l_t, vv * S * b / l_t
    ar_h, ar_v = 5.0, 1.5
    eta = 0.9  # tail dynamic-pressure ratio
    tau = 0.5  # flap effectiveness of a surface at about 35 % of its chord

    def helmbold(a):  # finite-wing lift slope, per rad
        return 2.0 * np.pi * a / (2.0 + math.sqrt(a**2 + 4.0))

    a_w, a_h, a_v = helmbold(AR), helmbold(ar_h), helmbold(ar_v)
    downwash = 2.0 * a_w / (np.pi * AR)
    CLa = a_w + eta * a_h * (s_h / S) * (1.0 - downwash)
    alpha_zero_lift = np.radians(-3.0)  # cambered sailplane airfoil
    CL0 = -CLa * alpha_zero_lift
    Cema = -static_margin * CLa
    cl_cruise = 2.0 * ac.weight / (ac.rho * S * 9.0**2)  # about 0.78 at 9 m/s
    alpha_cruise = (cl_cruise - CL0) / CLa
    Cem0 = -Cema * alpha_cruise  # trimmed with the elevator neutral at 9 m/s
    CLq = 2.0 * eta * a_h * vh
    Cemq = -2.0 * eta * a_h * vh * l_t / mac
    CLde = eta * a_h * (s_h / S) * tau
    Cemde = -eta * a_h * vh * tau

    # lateral-directional
    CYb = -eta * a_v * s_v / S
    Cenb = eta * a_v * vv - 0.005  # fin minus a fuselage share
    gamma = np.radians(dihedral_deg)
    Cellb = -(a_w * gamma / 6.0) * (1.0 + 2.0 * taper) / (1.0 + taper)
    Cellp = -(a_w / 12.0) * (1.0 + 3.0 * taper) / (1.0 + taper)
    Cenp = -cl_cruise / 8.0
    Cellr = cl_cruise / 4.0
    Cenr = -2.0 * eta * a_v * vv * l_t / b - ac.cd0 / 4.0
    CYr = 2.0 * eta * a_v * (s_v / S) * l_t / b
    CYp = 0.0
    # ailerons on 0.6..1.0 of the semispan: Cl_da = 2 a_w tau / (S b) int c(y) y dy
    y = np.linspace(0.6 * b / 2.0, b / 2.0, 201)
    chord = c_root * (1.0 - (1.0 - taper) * y / (b / 2.0))
    Cellda = 2.0 * a_w * 0.45 / (S * b) * np.trapezoid(chord * y, y)
    Cenda = -0.08 * cl_cruise * Cellda  # adverse yaw, halved by differential ailerons
    tau_r = 0.6  # full-height rudder, about half the fin chord
    CYdr = -eta * a_v * (s_v / S) * tau_r  # positive rudder yaws right: the fin's force points left
    Cendr = eta * a_v * vv * tau_r
    Celldr = -CYdr * 0.15 / b  # fin centre of pressure 0.15 m above the roll axis

    # inertia: wing as a tapered slender beam, tail at l_t, fuselage as a rod
    m_wing, m_tail, m_fuse = 1.6, 0.25, 0.6
    jx = m_wing * (b / 2.0) ** 2 / 3.0 * 0.75  # tapered: mass towards the root
    jy = m_tail * l_t**2 + m_fuse * 1.7**2 / 12.0
    jz = jx + jy
    k = inertia_scale

    return SixDofAirframe(
        name="SB-XC (estimated 6-DOF)", mass=ac.mass, jx=k * jx, jy=k * jy, jz=k * jz, jxz=0.0,
        area=S, span=b, mac=mac, oswald=e,
        CL0=CL0, CLa=CLa, CLq=CLq, CLde=CLde, CD0=ac.cd0, CDq=0.0,
        Cem0=Cem0, Cema=Cema, Cemq=Cemq, Cemde=Cemde,
        CYb=CYb, CYp=CYp, CYr=CYr, CYdr=CYdr,
        Cellb=Cellb, Cellp=Cellp, Cellr=Cellr, Cellda=Cellda, Celldr=Celldr,
        Cenb=Cenb, Cenp=Cenp, Cenr=Cenr, Cenda=Cenda, Cendr=Cendr,
        # control drag: the strip wing's 0.1 per (local delta C_L)^2, times each surface's local lift slope and
        # share of the reference area
        CDde2=0.1 * (a_h * tau) ** 2 * s_h / S, CDda2=0.1 * (a_w * 0.45) ** 2 * 0.3, CDdr2=0.1 * (a_v * tau_r) ** 2 * s_v / S,
        alpha_stall=_blend_midpoint(CL0, CLa, ac.cl_max, 50.0), rho=ac.rho,
        notes=dict(tail_arm=l_t, s_h=s_h, s_v=s_v, c_root=c_root, static_margin=static_margin,
                   dihedral_deg=dihedral_deg, a_w=a_w, cl_cruise=cl_cruise),
    )


def _blend_midpoint(cl0: float, cla: float, cl_max: float, m: float) -> float:
    """The plugin's ``alpha_stall`` is the MIDPOINT of its blend into the flat plate, where the lift is
    already half gone: placed at the linear curve's C_L max, the real maximum would fall near 0.8 instead
    of 1.2. This finds the midpoint whose blended curve peaks at ``cl_max``."""
    from scipy.optimize import brentq

    def peak(a_s):
        probe = SixDofAirframe.__new__(SixDofAirframe)
        object.__setattr__(probe, "M", m)
        object.__setattr__(probe, "alpha_stall", a_s)
        a = np.linspace(0.0, a_s + 0.2, 2001)
        sig = np.array([stall_blend(probe, x) for x in a])
        cl = (1 - sig) * (cl0 + cla * a) + sig * 2 * np.sin(a) ** 2 * np.cos(a)
        return cl.max() - cl_max

    return brentq(peak, (cl_max - cl0) / cla, (cl_max - cl0) / cla + 0.3)


# -- kinematics --------------------------------------------------------------------------------
def rotation(phi: float, theta: float, psi: float) -> np.ndarray:
    """Body (FRD) -> NED rotation matrix, Z-Y-X Euler angles."""
    cf, sf, ct, st, cp, sp = math.cos(phi), math.sin(phi), math.cos(theta), math.sin(theta), math.cos(psi), math.sin(psi)
    return np.array([
        [ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp],
        [ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp],
        [-st, sf * ct, cf * ct],
    ])


def enu_to_ned(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return np.stack([v[..., 1], v[..., 0], -v[..., 2]], axis=-1)


ned_to_enu = enu_to_ned  # the swap is its own inverse


# -- aerodynamics (port of gz-sim AdvancedLiftDrag) -------------------------------------------
@dataclass
class AirData:
    airspeed: float  # in the lift-drag plane, as the plugin
    alpha: float
    beta: float
    qbar: float


def air_data(af: SixDofAirframe, x: np.ndarray, wind_ned: np.ndarray) -> AirData:
    rot = rotation(x[PHI], x[THETA], x[PSI])
    ur, vr, wr = x[U:W + 1] - rot.T @ wind_ned
    speed = math.sqrt(ur * ur + wr * wr)
    return AirData(speed, math.atan2(wr, ur), math.atan2(vr, ur), 0.5 * af.rho * speed * speed)


def stall_blend(af: SixDofAirframe, alpha: float) -> float:
    """The plugin's sigmoid: 0 in the linear range, 1 beyond +-alpha_stall."""
    a = min(af.M * (alpha + af.alpha_stall), 50.0)
    b = min(-af.M * (alpha - af.alpha_stall), 50.0)
    ea, eb = math.exp(a), math.exp(b)
    return (1.0 + eb + ea) / ((1.0 + eb) * (1.0 + ea))


def aero_wrench(af: SixDofAirframe, x: np.ndarray, surfaces: np.ndarray, wind_ned: np.ndarray):
    """Aerodynamic force (body frame, N) and moment (body frame, N m), and the air data."""
    ad = air_data(af, x, wind_ned)
    if ad.airspeed < 1e-6:
        return np.zeros(3), np.zeros(3), ad
    de, da, dr = surfaces
    alpha, beta, qbar, speed = ad.alpha, ad.beta, ad.qbar, ad.airspeed
    p_hat = x[P] * af.span / (2.0 * speed)
    q_hat = x[Q] * af.mac / (2.0 * speed)
    r_hat = x[R] * af.span / (2.0 * speed)
    sigma = stall_blend(af, alpha)
    sa, ca = math.sin(alpha), math.cos(alpha)

    cl = (1.0 - sigma) * (af.CL0 + af.CLa * alpha) + sigma * 2.0 * math.copysign(1.0, alpha) * sa * sa * ca
    cl += af.CLde * de
    cd_fp = 2.0 / (1.0 + math.exp(af.CD_fp_k1 + af.CD_fp_k2 * max(af.aspect_ratio, 1.0 / af.aspect_ratio)))
    cd = ((1.0 - sigma) * (af.CD0 + cl * cl / (math.pi * af.aspect_ratio * af.oswald))
          + sigma * abs(cd_fp * (0.5 - 0.5 * math.cos(2.0 * alpha))))
    cd += af.CDde2 * de * de + af.CDda2 * da * da + af.CDdr2 * dr * dr
    cy = af.CYb * beta + af.CYdr * dr

    lift = qbar * af.area * (cl + af.CLq * q_hat)
    drag = qbar * af.area * (cd + af.CDq * q_hat)
    side = qbar * af.area * (cy + af.CYp * p_hat + af.CYr * r_hat)

    slope_stall = af.Cema if af.Cema_stall is None else af.Cema_stall
    if alpha > af.alpha_stall:
        cm = af.Cem0 + af.Cema * af.alpha_stall + slope_stall * (alpha - af.alpha_stall)
    elif alpha < -af.alpha_stall:
        cm = af.Cem0 - af.Cema * af.alpha_stall + slope_stall * (alpha + af.alpha_stall)
    else:
        cm = af.Cem0 + af.Cema * alpha
    cm += af.Cemde * de + af.Cemq * q_hat
    cell = af.Cellb * beta + af.Cellda * da + af.Celldr * dr + af.Cellp * p_hat + af.Cellr * r_hat
    cen = af.Cenb * beta + af.Cenda * da + af.Cendr * dr + af.Cenp * p_hat + af.Cenr * r_hat

    # stability frame -> body: lift along (sin a, 0, -cos a), drag along (-cos a, 0, -sin a)
    force = np.array([lift * sa - drag * ca, side, -lift * ca - drag * sa])
    moment = qbar * af.area * np.array([af.span * cell, af.mac * cm, af.span * cen])
    return force, moment, ad


# -- equations of motion -------------------------------------------------------------------------
def _gammas(af: SixDofAirframe):
    g = af.jx * af.jz - af.jxz**2
    return (af.jxz * (af.jx - af.jy + af.jz) / g, (af.jz * (af.jz - af.jy) + af.jxz**2) / g, af.jz / g,
            af.jxz / g, (af.jz - af.jx) / af.jy, af.jxz / af.jy, ((af.jx - af.jy) * af.jx + af.jxz**2) / g,
            af.jx / g)


def derivatives(af: SixDofAirframe, x: np.ndarray, surfaces: np.ndarray, thrust: float, wind_ned: np.ndarray):
    """``x_dot`` of the 12 rigid-body states, and the inertial (NED) acceleration."""
    force, moment, _ = aero_wrench(af, x, surfaces, wind_ned)
    phi, theta, psi = x[PHI], x[THETA], x[PSI]
    sf, cf, st, ct = math.sin(phi), math.cos(phi), math.sin(theta), math.cos(theta)
    force = force + np.array([thrust, 0.0, 0.0]) + af.mass * af.g * np.array([-st, sf * ct, cf * ct])
    u, v, w, p, q, r = x[U], x[V], x[W], x[P], x[Q], x[R]
    rot = rotation(phi, theta, psi)
    g1, g2, g3, g4, g5, g6, g7, g8 = _gammas(af)
    ell, em, en = moment
    xdot = np.empty(12)
    xdot[PN:PD + 1] = rot @ x[U:W + 1]
    xdot[U:W + 1] = np.array([r * v - q * w, p * w - r * u, q * u - p * v]) + force / af.mass
    xdot[PHI] = p + (q * sf + r * cf) * math.tan(theta)
    xdot[THETA] = q * cf - r * sf
    xdot[PSI] = (q * sf + r * cf) / ct
    xdot[P] = g1 * p * q - g2 * q * r + g3 * ell + g4 * en
    xdot[Q] = g5 * p * r - g6 * (p * p - r * r) + em / af.jy
    xdot[R] = g7 * p * q - g1 * q * r + g4 * ell + g8 * en
    return xdot, rot @ (force / af.mass)


# -- trim and linearization -------------------------------------------------------------------------
def trim_glide(af: SixDofAirframe, airspeed: float, thrust: float = 0.0):
    """Steady straight flight at ``airspeed`` in still air: ``(x0, u0)`` with every derivative but position at zero.

    Unknowns: alpha, elevator and path angle (thrust given). Lateral states are zero.
    """
    from scipy.optimize import least_squares

    def state(z):
        alpha, de, gamma = z
        x = np.zeros(12)
        x[PD] = -400.0
        x[U], x[W] = airspeed * math.cos(alpha), airspeed * math.sin(alpha)
        x[THETA] = alpha + gamma
        return x, np.array([de, 0.0, 0.0])

    def residual(z):
        x, s = state(z)
        xdot, _ = derivatives(af, x, s, thrust, np.zeros(3))
        return np.array([xdot[U], xdot[W], xdot[Q]])

    cl = af.weight / (0.5 * af.rho * airspeed**2 * af.area)
    sol = least_squares(residual, [(cl - af.CL0) / af.CLa, 0.0, -0.04], xtol=1e-12, ftol=1e-12)
    x0, s0 = state(sol.x)
    return x0, np.concatenate([s0, [thrust]])


def linearize(af: SixDofAirframe, x0: np.ndarray, u0: np.ndarray, eps: float = 1e-6):
    """``A = df/dx`` (12 x 12) and ``B = df/du`` (12 x 4) by central differences, still air."""

    def f(x, u):
        return derivatives(af, x, u[:3], u[3], np.zeros(3))[0]

    a = np.stack([(f(x0 + eps * e, u0) - f(x0 - eps * e, u0)) / (2 * eps) for e in np.eye(12)], axis=1)
    b = np.stack([(f(x0, u0 + eps * e) - f(x0, u0 - eps * e)) / (2 * eps) for e in np.eye(4)], axis=1)
    return a, b


# -- autopilot -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AutopilotGains:
    """Inner loops taking the point mass's commands (C_L, bank, thrust), PX4-style airspeed-scaled rate loops.

    Pitch: C_L -> angle of attack -> pitch rate -> elevator, with the trim
    elevator as feedforward; an integrator on the measured lift-coefficient
    error (accelerometer and air data) corrects alpha where the lift curve
    bends before stall, and alpha stays a degree below the angle of maximum lift.
    Envelope protections: load factor at least 0.5, a 1.8 g pull-up beyond
    30 deg nose down, bank within 45 deg (the point-mass tracker can command a
    push-over it never has to recover from; the rigid body can enter a dive
    the tracker's flat map cannot pull out of). Roll: bank
    -> roll rate (limited, like the point mass's tracker) -> aileron. Yaw:
    the coordinated-turn yaw rate and zero sideslip -> rudder. Gains are
    bandwidths (1/s); dividing by ``qbar S c Cm_de`` and friends turns a
    commanded angular acceleration into a deflection.
    """

    k_alpha: float = 6.0
    k_cl_i: float = 3.0  # 1/s, integral of the lift-coefficient error
    alpha_margin: float = np.radians(1.0)  # alpha protection below the angle of maximum lift
    cl_protect_fraction: float = 0.9  # ... and at most the alpha of this fraction of C_L max (the tracker's limit)
    k_q: float = 10.0
    k_phi: float = 2.5
    k_phi_i: float = 0.3
    k_p: float = 8.0
    roll_rate_max: float = np.radians(20.0)  # with margin: full aileron rolls the SB-XC estimate at ~32 deg/s at 9 m/s
    k_r: float = 4.0
    k_beta: float = 6.0  # 1/s^2 of yaw acceleration per rad of sideslip
    n_min: float = 0.5  # load-factor floor
    pitch_down_max: float = np.radians(30.0)  # beyond it, pull up at n_pull_up
    n_pull_up: float = 1.8
    bank_max: float = np.radians(45.0)


def sixdof_tracker(base):
    """The path tracker retuned for the 6-DOF airframe: cross-track gains 0.05 / 0.3 and airspeed gain 0.4
    (the point mass's 0.3 / 1.0 / 0.8 drive a 20 deg/s roll into a lateral oscillation that ends in a spiral;
    tuned on weather-family situations 600-615, every flight held)."""
    return replace(base, k_p=0.05, k_v=0.3, k_speed=0.4)


class SixDofPlant:
    """A 6-DOF airframe flown by its autopilot, stepped by the path tracker of :mod:`geneticwing.simulation`.

    The tracker hands over ``(C_L, bank, thrust)`` every tracker step; the
    autopilot runs at every substep (``substep`` seconds, RK4). Position and
    velocity come back in the tracker's ENU frame, ground-relative.
    """

    def __init__(self, airframe: SixDofAirframe, gains: AutopilotGains = AutopilotGains(), substep: float = 0.01):
        self.af, self.gains, self.substep = airframe, gains, substep
        self.x: np.ndarray | None = None  # 15: 12 rigid-body states + 3 servo positions
        self.integral = np.zeros(2)  # alpha, bank
        # protect the same C_L as the point mass's tracker (a fraction of C_L max), a degree below max lift at most
        a = np.linspace(0.0, airframe.alpha_cl_max, 2001)
        protected = float(a[np.searchsorted(airframe.static_cl(a), gains.cl_protect_fraction * float(airframe.static_cl(airframe.alpha_cl_max)))])
        self.alpha_limit = min(protected, airframe.alpha_cl_max - gains.alpha_margin)
        self.log: dict[str, list] = {k: [] for k in ("t", "alpha", "beta", "phi", "p", "q", "r", "de", "da", "dr",
                                                     "thrust", "saturated", "cl_cmd", "cl")}

    # -- start ----------------------------------------------------------------------------------
    def begin(self, p_enu, v_enu, wind, t: float, cl: float, bank: float) -> None:
        """Start from a point-mass state (first segment), or carry on if already flying from there."""
        p_ned = enu_to_ned(p_enu)
        if self.x is not None and np.linalg.norm(self.x[PN:PD + 1] - p_ned) < 1e-6:
            return
        af = self.af
        w_ned = enu_to_ned(wind(np.asarray(p_enu, float), t))
        v_air = enu_to_ned(v_enu) - w_ned
        speed = float(np.linalg.norm(v_air))
        psi = math.atan2(v_air[1], v_air[0])
        gamma = math.asin(np.clip(-v_air[2] / speed, -1.0, 1.0))
        alpha = (cl - af.CL0) / af.CLa
        theta = gamma + alpha * math.cos(bank)
        x = np.zeros(15)
        x[PN:PD + 1] = p_ned
        x[PHI], x[THETA], x[PSI] = bank, theta, psi
        rot = rotation(bank, theta, psi)
        x[U:W + 1] = rot.T @ (v_air + w_ned)  # ground (inertial) velocity in body axes
        omega = af.g * math.tan(bank) / max(speed, 1.0)  # coordinated turn rate
        x[P], x[Q], x[R] = -omega * math.sin(theta), omega * math.sin(bank) * math.cos(theta), omega * math.cos(bank) * math.cos(theta)
        x[DE] = self._elevator_feedforward(alpha, x[Q] * af.mac / (2 * speed))
        self.x = x
        self.integral[:] = 0.0

    def _elevator_feedforward(self, alpha: float, q_hat: float) -> float:
        af = self.af
        return -(af.Cem0 + af.Cema * alpha + af.Cemq * q_hat) / af.Cemde

    # -- autopilot ----------------------------------------------------------------------------------
    def _surface_commands(self, x, cl_cmd, bank_cmd, wind_ned, dt):
        af, k = self.af, self.gains
        ad = air_data(af, x, wind_ned)
        speed = max(ad.airspeed, 3.0)
        qbar = max(ad.qbar, 1.0)
        phi, theta = x[PHI], x[THETA]
        omega = af.g * math.tan(phi) / speed
        # envelope protections, as an autopilot has: no push below n_min, and a pull-up beyond the pitch-down limit
        cl_cmd = max(cl_cmd, k.n_min * af.weight / (qbar * af.area))
        if theta < -k.pitch_down_max:
            cl_cmd = max(cl_cmd, k.n_pull_up * af.weight / (qbar * af.area))
        bank_cmd = float(np.clip(bank_cmd, -k.bank_max, k.bank_max))
        # pitch: the lift coefficient the accelerometer and the air data measure, held by trimming alpha
        force, _, _ = aero_wrench(af, x[:12], x[12:], wind_ned)
        sa, ca = math.sin(ad.alpha), math.cos(ad.alpha)
        cl_meas = (force[0] * sa - force[2] * ca) / (qbar * af.area)
        self._cl = (cl_cmd, cl_meas)
        alpha_free = (cl_cmd - af.CL0) / af.CLa + self.integral[0] / af.CLa
        if not (alpha_free >= self.alpha_limit and cl_cmd > cl_meas):  # anti-windup at the alpha protection
            self.integral[0] = np.clip(self.integral[0] + k.k_cl_i * (cl_cmd - cl_meas) * dt, -0.3, 0.3)
        alpha_cmd = min((cl_cmd - af.CL0) / af.CLa + self.integral[0] / af.CLa, self.alpha_limit)  # alpha protection
        err_a = alpha_cmd - ad.alpha
        # the pitch rate that keeps alpha constant: the flight path's curvature under the commanded lift
        # (in a level turn it is the turn's own pitch rate, omega sin(phi))
        q_ff = (qbar * af.area * cl_cmd / af.mass - af.g * math.cos(theta) * math.cos(phi)) / speed
        q_cmd = q_ff + k.k_alpha * err_a
        if ad.alpha > self.alpha_limit:  # the protection overrides the turn's feedforward
            q_cmd = min(q_cmd, 2.0 * k.k_alpha * (self.alpha_limit - ad.alpha))
        de = (self._elevator_feedforward(alpha_cmd, q_ff * af.mac / (2 * speed))
              + af.jy * k.k_q * (q_cmd - x[Q]) / (qbar * af.area * af.mac * af.Cemde))
        # roll
        err_phi = bank_cmd - phi
        self.integral[1] = np.clip(self.integral[1] + err_phi * dt, -0.2, 0.2)
        p_cmd = np.clip(k.k_phi * err_phi + k.k_phi_i * self.integral[1], -k.roll_rate_max, k.roll_rate_max)
        p_cmd += -omega * math.sin(theta)
        to_hat = af.span / (2.0 * speed)
        # rate feedforward (PX4's FF gain): the deflection that balances roll damping at the commanded rate
        da = (-af.Cellp * p_cmd * to_hat / af.Cellda
              + af.jx * k.k_p * (p_cmd - x[P]) / (qbar * af.area * af.span * af.Cellda))
        # yaw: coordinated-turn yaw rate and zero sideslip, the aileron's adverse yaw and the yaw damping fed forward
        r_cmd = omega * math.cos(phi) * math.cos(theta)
        dr = ((-af.Cenr * r_cmd * to_hat - af.Cenp * x[P] * to_hat - af.Cenda * da) / af.Cendr
              + af.jz * (k.k_r * (r_cmd - x[R]) + k.k_beta * ad.beta) / (qbar * af.area * af.span * af.Cendr))
        cmd = np.array([de, da, dr])
        lim = np.asarray(af.deflection_max)
        return np.clip(cmd, -lim, lim), bool(np.any(np.abs(cmd) > lim))

    def _rates(self, x, cmd, thrust, wind_ned):
        af = self.af
        xdot = np.empty(15)
        xdot[:12], acc = derivatives(af, x[:12], x[12:], thrust, wind_ned)
        xdot[12:] = np.clip((cmd - x[12:]) / af.servo_time_constant, -af.servo_rate_max, af.servo_rate_max)
        return xdot, acc

    # -- one tracker step ---------------------------------------------------------------------------
    def step(self, wind, t: float, dt: float, cl_cmd: float, bank_cmd: float, thrust: float):
        """Fly ``dt`` seconds; returns ``(p_enu, v_enu, a_enu at the start)`` of the new state."""
        n = max(1, int(round(dt / self.substep)))
        h = dt / n
        acc0 = None

        def wind_ned_at(x, time):
            return enu_to_ned(wind(ned_to_enu(x[PN:PD + 1]), time))

        for i in range(n):
            ti = t + i * h
            x = self.x
            cmd, saturated = self._surface_commands(x, cl_cmd, bank_cmd, wind_ned_at(x, ti), h)
            k1, a1 = self._rates(x, cmd, thrust, wind_ned_at(x, ti))
            x2 = x + 0.5 * h * k1
            k2, _ = self._rates(x2, cmd, thrust, wind_ned_at(x2, ti + 0.5 * h))
            x3 = x + 0.5 * h * k2
            k3, _ = self._rates(x3, cmd, thrust, wind_ned_at(x3, ti + 0.5 * h))
            x4 = x + h * k3
            k4, _ = self._rates(x4, cmd, thrust, wind_ned_at(x4, ti + h))
            self.x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            if i == 0:
                acc0 = a1
                ad = air_data(self.af, x, wind_ned_at(x, ti))
                for key, val in (("t", ti), ("alpha", ad.alpha), ("beta", ad.beta), ("phi", x[PHI]), ("p", x[P]),
                                 ("q", x[Q]), ("r", x[R]), ("de", x[DE]), ("da", x[DA]), ("dr", x[DR]),
                                 ("thrust", thrust), ("saturated", saturated), ("cl_cmd", self._cl[0]),
                                 ("cl", self._cl[1])):
                    self.log[key].append(val)
        return self.position_enu(), self.velocity_enu(), ned_to_enu(acc0)

    def position_enu(self) -> np.ndarray:
        return ned_to_enu(self.x[PN:PD + 1])

    def velocity_enu(self) -> np.ndarray:
        rot = rotation(self.x[PHI], self.x[THETA], self.x[PSI])
        return ned_to_enu(rot @ self.x[U:W + 1])

    def summary(self) -> dict[str, float]:
        lg = {k: np.asarray(v, dtype=float) for k, v in self.log.items()}
        if lg["t"].size == 0:
            return {}
        return dict(alpha_max_deg=float(np.degrees(lg["alpha"].max())),
                    beta_max_deg=float(np.degrees(np.abs(lg["beta"]).max())),
                    roll_rate_max_dps=float(np.degrees(np.abs(lg["p"]).max())),
                    aileron_max_deg=float(np.degrees(np.abs(lg["da"]).max())),
                    elevator_max_deg=float(np.degrees(np.abs(lg["de"]).max())),
                    rudder_max_deg=float(np.degrees(np.abs(lg["dr"]).max())),
                    saturated_fraction=float(lg["saturated"].mean()))
