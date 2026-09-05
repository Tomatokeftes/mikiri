"""Camera geometry on the court plane: height recovery, bias correction, uncertainty.

A ground-plane homography alone answers "where on the floor does this pixel
land". It cannot answer "how high was that" and it silently mis-places any
point that is *not* on the floor. Both gaps close once the camera's position
is known, and the court hands us the measurement for free: the net posts are
vertical segments of known height (1.55 m) standing at known court points, so
clicking a post base and its top pins the camera down.

Three things follow, in increasing order of usefulness:

1. **Off-plane bias correction.** Ankle keypoints sit about 7.5 cm above the
   floor. Mapped as if they were on it they land *away* from the camera by
   ``z * d / (h - z)``, which is 92 cm at the far baseline for a 1.6 m tripod:
   far larger than keypoint jitter, and entirely systematic.
2. **Heights in metres.** Invert the same relation and an off-plane point
   whose horizontal position is known gives its height: jump clearance, shuttle
   altitude, racket contact height.
3. **Honest error bars.** First-order propagation of pixel noise through the
   homography gives a position covariance per point, which is strongly
   anisotropic: at the far baseline the error ellipse is several times longer
   along the court than across it.

Geometry and notation. The camera sits at ``C = (Gx, Gy, h)`` in court
coordinates; ``G = (Gx, Gy)`` is its ground point (the nadir) and ``h`` its
height. A world point at height ``z`` above court point ``B`` is seen along the
ray ``C -> (B, z)``; extending that ray to the floor gives the point ``M`` a
ground homography reports::

    M = B + lam * (B - G)        with lam = z / (h - z)

which is the single relation this module solves forwards (bias), backwards
(height) and for the camera itself (calibration).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import least_squares

#: Height of the net posts above the floor (BWF Laws of Badminton, 1.55 m).
NET_POST_HEIGHT = 1.55
#: Height of the net at the centre of the court (BWF, 1.524 m).
NET_CENTRE_HEIGHT = 1.524
#: Ankle joint centre above the floor for an adult, metres. The bias this
#: causes is proportional to it, so it is a tunable rather than a constant.
ANKLE_HEIGHT = 0.075
#: Default 1-sigma jitter of a pose keypoint, in pixels at the clip's own
#: resolution. RTMPose ankles on 1080p footage sit around this.
SIGMA_KEYPOINT_PX = 2.5
#: Default 1-sigma error of a hand-clicked court landmark, in pixels.
SIGMA_CLICK_PX = 3.0
#: Ellipse scale for 95% coverage in two dimensions: ``sqrt(chi2(0.95, 2))``.
#: A one-sigma ellipse covers only 39.35% of a 2-D Gaussian, so quoting "1
#: sigma" as if it were the familiar 68% figure overstates the accuracy by a
#: wide margin. Reported error ellipses use this instead.
NSIGMA_95 = 2.4477468306808364
#: Two-sided 95% factor for a SCALAR quantity. Distinct from NSIGMA_95: a
#: one-dimensional interval needs 1.96 sigma, and using the 2-D ellipse factor
#: on a scalar over-states the interval by a quarter.
Z_95_SCALAR = 1.959963984540054
#: How far up the court a segment top may map before the recovery is declared
#: degenerate. A net-post top belongs a few court lengths away at most; when it
#: maps hundreds of metres out, it is sitting on the horizon and the camera is
#: at about post height. Generous, because the honest failure is unmistakable.
MAX_OFFPLANE_REACH_M = 60.0
#: Largest disagreement between vertical segments, in metres, that still counts
#: as a consistent camera recovery. Two net posts over-determine the fit, so
#: their residual is a real check: a single post top clicked 30 px off produces
#: a metre of disagreement and a camera height 7% low.
MAX_CAMERA_RESIDUAL_M = 0.35


class CameraGeometry(BaseModel):
    """Where the camera is, in court coordinates and metres.

    ``ground_point`` is the camera's nadir (directly below it) in the court
    frame; it is routinely outside the court and often behind the baseline, so
    negative and out-of-range values are expected and not an error.
    """

    ground_point: list[float] = Field(..., description="Camera nadir (x, y) in court metres")
    height_m: float = Field(..., description="Camera height above the floor in metres")
    source: str = Field("vertical_pair", description="How it was obtained")
    residual_m: float | None = Field(
        None, description="RMS disagreement between the vertical segments used, in metres"
    )

    def is_plausible(self) -> tuple[bool, str]:
        """Sanity-check the recovered geometry against what a real camera can be.

        Returns ``(ok, reason)``; ``reason`` is empty when ``ok``.
        """
        h = self.height_m
        if not math.isfinite(h):
            return False, "camera height is not finite"
        if h <= NET_POST_HEIGHT + 0.05:
            return False, f"camera height {h:.2f} m is at or below the net posts"
        if h > 40.0:
            return False, f"camera height {h:.1f} m is implausibly high"
        g = np.asarray(self.ground_point, dtype=float)
        if not np.isfinite(g).all():
            return False, "camera ground point is not finite"
        if np.abs(g).max() > 100.0:
            return False, "camera ground point is implausibly far from the court"
        if self.residual_m is not None and self.residual_m > MAX_CAMERA_RESIDUAL_M:
            return (
                False,
                f"the net posts disagree by {self.residual_m:.2f} m; check the clicked tops",
            )
        return True, ""


@dataclass
class VerticalSegment:
    """A vertical object of known height whose base sits at a known court point.

    ``base_court`` is the base in court metres, ``top_image`` the top in image
    pixels, ``height_m`` the true vertical length. Net posts are the obvious
    instance; a doorway, a chair back or a person of known height standing still
    all work identically.
    """

    base_court: tuple[float, float]
    top_image: tuple[float, float]
    height_m: float = NET_POST_HEIGHT


def _lam(z: float, h: float) -> float:
    """Ray-extension factor ``z / (h - z)`` for height ``z`` under camera height ``h``."""
    return z / (h - z)


def project_offplane(
    base_court: np.ndarray, z: float | np.ndarray, camera: CameraGeometry
) -> np.ndarray:
    """Where points at height ``z`` above ``base_court`` land under a floor homography.

    This is the forward direction of the module's central relation: the answer
    is always further from the camera than the true base.

    Args:
        base_court: ``(n, 2)`` true horizontal positions in court metres.
        z: Height above the floor in metres, scalar or ``(n,)``.
        camera: Recovered camera geometry.

    Returns:
        ``(n, 2)`` positions a ground homography would report, in court metres.
    """
    b = np.asarray(base_court, dtype=float).reshape(-1, 2)
    g = np.asarray(camera.ground_point, dtype=float)
    zz = np.asarray(z, dtype=float).reshape(-1, 1)
    lam = zz / (camera.height_m - zz)
    return b + lam * (b - g)


def correct_offplane(
    mapped_court: np.ndarray, z: float | np.ndarray, camera: CameraGeometry
) -> np.ndarray:
    """Undo the off-plane bias: recover the true horizontal position from a mapped one.

    The inverse of :func:`project_offplane`. Applying it to mapped ankle points
    with ``z = ANKLE_HEIGHT`` removes the systematic far-court stretch.

    Args:
        mapped_court: ``(n, 2)`` positions reported by the ground homography.
        z: True height above the floor of the mapped feature, metres.
        camera: Recovered camera geometry.

    Returns:
        ``(n, 2)`` corrected horizontal positions in court metres.
    """
    m = np.asarray(mapped_court, dtype=float).reshape(-1, 2)
    g = np.asarray(camera.ground_point, dtype=float)
    zz = np.asarray(z, dtype=float).reshape(-1, 1)
    # M = G + (M - G); B = G + (1 - z/h) * (M - G)
    shrink = 1.0 - zz / camera.height_m
    return g + shrink * (m - g)


def height_from_offset(
    mapped_court: np.ndarray, true_horizontal: np.ndarray, camera: CameraGeometry
) -> np.ndarray:
    """Height above the floor implied by the gap between mapped and true positions.

    The backwards direction: when a point's horizontal position is known by
    other means (a smoothed trajectory through a jump, a shuttle constrained to
    a vertical plane), how far the ground homography over-shoots gives its
    height directly.

    Args:
        mapped_court: ``(n, 2)`` positions reported by the ground homography.
        true_horizontal: ``(n, 2)`` known true horizontal positions, court metres.
        camera: Recovered camera geometry.

    Returns:
        ``(n,)`` heights in metres, ``nan`` where the geometry is degenerate.
        Values are clipped below at 0: a point cannot be under the floor.
    """
    m = np.asarray(mapped_court, dtype=float).reshape(-1, 2)
    b = np.asarray(true_horizontal, dtype=float).reshape(-1, 2)
    g = np.asarray(camera.ground_point, dtype=float)
    dm = m - g
    db = b - g
    denom = np.einsum("ij,ij->i", dm, dm)
    with np.errstate(divide="ignore", invalid="ignore"):
        # z / h = 1 - |B - G| / |M - G|, projected onto the ray for stability
        ratio = np.einsum("ij,ij->i", db, dm) / denom
        z = camera.height_m * (1.0 - ratio)
    z[~np.isfinite(z)] = np.nan
    return np.maximum(z, 0.0)


def camera_from_verticals(segments: list[VerticalSegment], H: np.ndarray) -> CameraGeometry:
    """Recover camera ground point and height from vertical segments of known height.

    Each segment contributes the two-component relation ``M - B = lam (B - G)``
    where ``M`` is the segment top mapped through the floor homography. Two net
    posts give four equations for three unknowns, so the fit is overdetermined
    and its residual is a real check on the calibration.

    When every segment shares one height the problem is linear in
    ``(lam, lam * G)`` and is solved directly; that solution then seeds a
    non-linear refinement which also handles mixed heights.

    Args:
        segments: At least two vertical segments; more is better.
        H: Image-to-court homography (3, 3), as from ``court.compute_homography``.

    Returns:
        The recovered :class:`CameraGeometry`, with ``residual_m`` set.

    Raises:
        ValueError: Fewer than two segments, or a degenerate configuration
            (all bases coincident, or a fit implying a camera at or below the
            segment tops).
    """
    if len(segments) < 2:
        raise ValueError("need at least two vertical segments to recover the camera")
    H = np.asarray(H, dtype=float)
    from mikiri.court import image_to_court  # local import: court does not import this module

    bases = np.array([s.base_court for s in segments], dtype=float)
    tops_img = np.array([s.top_image for s in segments], dtype=float)
    heights = np.array([s.height_m for s in segments], dtype=float)
    if not np.isfinite(bases).all() or not np.isfinite(tops_img).all():
        raise ValueError("vertical segments contain non-finite coordinates")
    if np.ptp(bases, axis=0).max() < 1e-6:
        raise ValueError("vertical segment bases are coincident")
    mapped = image_to_court(H, tops_img)
    if not np.isfinite(mapped).all():
        raise ValueError("a segment top maps behind the horizon; check the clicked points")

    # A vertical's top crosses the ground plane's horizon exactly when its
    # height equals the camera's. So as the camera drops toward the segment
    # height, the top approaches the horizon and its mapped court position runs
    # away to infinity: at 18 px from the horizon a net-post top maps 200 m up
    # the court, and a pixel of click error moves it by metres. The method is
    # not merely imprecise there, it is unusable -- and blaming the clicks would
    # send the user back to re-click points that were never the problem.
    reach = np.hypot(*(mapped - bases).T)
    if (reach > MAX_OFFPLANE_REACH_M).any():
        worst = float(np.max(reach))
        tallest = float(heights.max())
        raise ValueError(
            f"the camera is too close to {tallest:.2f} m for this to work: a segment top "
            f"maps {worst:.0f} m up the court, which means it sits within a few pixels of "
            "the horizon. Film from higher than the net posts by a clear margin, or leave "
            "the camera unset and lose only the height-dependent measurements"
        )

    # Linear seed, exact when all heights agree: unknowns u = lam, v = lam * G.
    rows, rhs = [], []
    for b, m in zip(bases, mapped):
        rows.append([b[0], -1.0, 0.0])
        rhs.append(m[0] - b[0])
        rows.append([b[1], 0.0, -1.0])
        rhs.append(m[1] - b[1])
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    u = float(sol[0])
    z0 = float(heights.mean())
    seed_h = z0 * (u + 1.0) / u if math.isfinite(u) and abs(u) > 1e-9 else float("nan")
    seed_g = np.array([sol[1] / u, sol[2] / u]) if math.isfinite(seed_h) else np.zeros(2)
    if not math.isfinite(seed_h) or seed_h <= heights.max() or not np.isfinite(seed_g).all():
        # The linear fit put the camera below the segment tops (or nowhere at
        # all); start the refinement from a plausible spot instead of failing.
        seed_h = float(heights.max()) * 2.0
        seed_g = bases.mean(axis=0) - np.array([0.0, 6.0])

    def residuals(p: np.ndarray) -> np.ndarray:
        g = p[:2]
        h = p[2]
        lam = heights / (h - heights)
        pred = bases + lam[:, None] * (bases - g)
        return (pred - mapped).ravel()

    fit = least_squares(
        residuals,
        x0=np.array([seed_g[0], seed_g[1], seed_h]),
        # Keep the camera strictly above the tallest segment.
        bounds=(
            np.array([-200.0, -200.0, float(heights.max()) + 1e-3]),
            np.array([200.0, 200.0, 200.0]),
        ),
        method="trf",
    )
    g = fit.x[:2]
    h = float(fit.x[2])
    rms = float(np.sqrt(np.mean(fit.fun**2))) if fit.fun.size else 0.0
    cam = CameraGeometry(
        ground_point=[float(g[0]), float(g[1])],
        height_m=h,
        source="vertical_pair" if len(segments) == 2 else "vertical_multi",
        residual_m=round(rms, 4),
    )
    ok, why = cam.is_plausible()
    if not ok:
        raise ValueError(f"recovered camera is implausible: {why}")
    return cam


def camera_from_height(segment: VerticalSegment, H: np.ndarray, height_m: float) -> CameraGeometry:
    """Recover the camera ground point from one vertical segment and a known height.

    One segment gives two equations for three unknowns; supplying the camera
    height (a tape measure, or the tripod's own scale) closes the system. Less
    accurate than two segments, but available when only one post is in frame.

    Args:
        segment: A single vertical segment of known height.
        H: Image-to-court homography (3, 3).
        height_m: Camera height above the floor, metres.

    Returns:
        The recovered :class:`CameraGeometry` with ``source="known_height"``.

    Raises:
        ValueError: The height is not above the segment, or the mapping is degenerate.
    """
    if not math.isfinite(height_m) or height_m <= segment.height_m:
        raise ValueError("camera height must be finite and above the segment top")
    from mikiri.court import image_to_court

    m = image_to_court(np.asarray(H, dtype=float), np.array([segment.top_image], dtype=float))[0]
    if not np.isfinite(m).all():
        raise ValueError("the segment top maps behind the horizon")
    b = np.asarray(segment.base_court, dtype=float)
    lam = _lam(segment.height_m, height_m)
    g = b - (m - b) / lam
    cam = CameraGeometry(
        ground_point=[float(g[0]), float(g[1])],
        height_m=float(height_m),
        source="known_height",
        residual_m=0.0,
    )
    ok, why = cam.is_plausible()
    if not ok:
        raise ValueError(f"recovered camera is implausible: {why}")
    return cam


def position_jacobian(H: np.ndarray, image_points: np.ndarray) -> np.ndarray:
    """Jacobian of the image-to-court map at each image point.

    For ``X = a/w``, ``Y = b/w`` with ``a, b, w`` the three rows of ``H`` applied
    to the homogeneous pixel, the derivative is::

        J = (1/w) * [[h00 - X*h20, h01 - X*h21],
                     [h10 - Y*h20, h11 - Y*h21]]

    Args:
        H: Image-to-court homography (3, 3).
        image_points: ``(n, 2)`` pixel coordinates.

    Returns:
        ``(n, 2, 2)`` Jacobians in metres per pixel; ``nan`` where ``w`` vanishes
        (points on the horizon).
    """
    H = np.asarray(H, dtype=float)
    pts = np.asarray(image_points, dtype=float).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    a = hom @ H[0]
    b = hom @ H[1]
    w = hom @ H[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        X = a / w
        Y = b / w
        J = np.empty((pts.shape[0], 2, 2))
        J[:, 0, 0] = H[0, 0] - X * H[2, 0]
        J[:, 0, 1] = H[0, 1] - X * H[2, 1]
        J[:, 1, 0] = H[1, 0] - Y * H[2, 0]
        J[:, 1, 1] = H[1, 1] - Y * H[2, 1]
        J /= w[:, None, None]
    J[~np.isfinite(J).all(axis=(1, 2))] = np.nan
    return J


def position_covariance(
    H: np.ndarray,
    image_points: np.ndarray,
    sigma_px: float = SIGMA_KEYPOINT_PX,
    calibration: tuple[np.ndarray, np.ndarray] | None = None,
    sigma_click_px: float = SIGMA_CLICK_PX,
    samples: int = 128,
    seed: int = 0,
) -> np.ndarray:
    """Court-position covariance for each image point, in square metres.

    Two independent contributions are summed:

    * **Keypoint jitter**, propagated to first order as ``sigma^2 J J^T``. This
      is exact to the precision that matters (checked against Monte Carlo) and
      is what makes the far court's error ellipse long and thin.
    * **Calibration error**, from the clicked landmarks. There is no tidy closed
      form once the homography is refitted, so it is sampled: the clicked image
      points are perturbed ``samples`` times and the spread of the resulting
      court positions is measured. Deterministic for a given ``seed``.

    Args:
        H: Image-to-court homography (3, 3).
        image_points: ``(n, 2)`` pixel coordinates to evaluate.
        sigma_px: 1-sigma keypoint jitter in pixels.
        calibration: ``(image_points, court_points)`` used to fit ``H``. When
            given, calibration error is included; when ``None`` it is omitted
            and the result is a lower bound.
        sigma_click_px: 1-sigma error of each clicked landmark, pixels.
        samples: Monte-Carlo draws for the calibration term.
        seed: Seed for those draws, so results are reproducible.

    Returns:
        ``(n, 2, 2)`` covariance matrices in m^2.
    """
    import cv2

    pts = np.asarray(image_points, dtype=float).reshape(-1, 2)
    J = position_jacobian(H, pts)
    cov = (sigma_px**2) * (J @ np.swapaxes(J, 1, 2))

    if calibration is not None and samples > 0 and sigma_click_px > 0:
        src = np.asarray(calibration[0], dtype=float).reshape(-1, 2)
        dst = np.asarray(calibration[1], dtype=float).reshape(-1, 2)
        if src.shape[0] >= 4 and src.shape == dst.shape:
            from mikiri.court import image_to_court

            rng = np.random.default_rng(seed)
            base = image_to_court(H, pts)
            draws = np.empty((samples, pts.shape[0], 2))
            kept = 0
            for _ in range(samples):
                jitter = src + rng.normal(0.0, sigma_click_px, src.shape)
                Hs, _ = cv2.findHomography(jitter, dst, method=0)
                if Hs is None or not np.isfinite(Hs).all() or abs(Hs[2, 2]) < 1e-12:
                    continue
                mapped = image_to_court(Hs / Hs[2, 2], pts)
                if not np.isfinite(mapped).all():
                    continue
                draws[kept] = mapped
                kept += 1
            if kept >= 8:
                d = draws[:kept] - base[None, :, :]
                # Covariance about the unperturbed mapping, not the sample mean:
                # calibration error is a bias we do not get to re-centre away.
                cal = np.einsum("kni,knj->nij", d, d) / kept
                cov = cov + cal
    return cov


def error_ellipse(
    cov: np.ndarray, nsigma: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Semi-axes and orientation of the error ellipses of ``(n, 2, 2)`` covariances.

    Args:
        cov: ``(n, 2, 2)`` covariance matrices in m^2.
        nsigma: Scale factor; 1.0 gives the 1-sigma ellipse, 2.45 roughly 95%
            for two degrees of freedom.

    Returns:
        ``(semi_major_m, semi_minor_m, angle_rad)``, each ``(n,)``. The angle is
        the major axis measured from the court ``x`` axis (across the court),
        so an angle near +/- pi/2 means the error is mostly along the court.
    """
    c = np.asarray(cov, dtype=float).reshape(-1, 2, 2)
    out_major = np.full(c.shape[0], np.nan)
    out_minor = np.full(c.shape[0], np.nan)
    out_angle = np.full(c.shape[0], np.nan)
    ok = np.isfinite(c).all(axis=(1, 2))
    if ok.any():
        sym = 0.5 * (c[ok] + np.swapaxes(c[ok], 1, 2))
        vals, vecs = np.linalg.eigh(sym)
        vals = np.maximum(vals, 0.0)
        out_major[ok] = nsigma * np.sqrt(vals[:, 1])
        out_minor[ok] = nsigma * np.sqrt(vals[:, 0])
        out_angle[ok] = np.arctan2(vecs[:, 1, 1], vecs[:, 0, 1])
    return out_major, out_minor, out_angle


def offplane_bias(court_points: np.ndarray, z: float, camera: CameraGeometry) -> np.ndarray:
    """Magnitude of the off-plane displacement at each court point, in metres.

    What a feature at height ``z`` would be displaced by if mapped as if on the
    floor. Reported so the size of the correction stays visible even once the
    correction is applied.

    Args:
        court_points: ``(n, 2)`` true horizontal positions, court metres.
        z: Height above the floor, metres.
        camera: Recovered camera geometry.

    Returns:
        ``(n,)`` displacement magnitudes in metres.
    """
    b = np.asarray(court_points, dtype=float).reshape(-1, 2)
    g = np.asarray(camera.ground_point, dtype=float)
    lam = _lam(z, camera.height_m)
    return lam * np.hypot(b[:, 0] - g[0], b[:, 1] - g[1])


#: The three ways a camera can look at a court, as far as 2-D analysis cares.
VIEW_SIDE_ON = "side-on"
VIEW_REAR = "rear"
VIEW_OBLIQUE = "oblique"
VIEW_CLASSES = (VIEW_SIDE_ON, VIEW_REAR, VIEW_OBLIQUE)

#: Half-width of the cone, in degrees, within which a view counts as square-on
#: to one of the court axes. Outside both cones the view is oblique and the
#: image-plane angles are foreshortened by an amount nothing can recover.
VIEW_CONE_DEG = 30.0

#: Which view each family of measurements needs, and why. Anything absent from
#: this table is view-robust: it comes from the homography and the recovered
#: camera rather than from angles measured in the image, so it survives any
#: viewpoint. That is the whole argument for leaning on the court geometry.
VIEW_REQUIREMENTS: dict[str, dict] = {
    "trunk_lean": {
        "view": VIEW_SIDE_ON,
        "why": "Lean toward or away from the net is a rotation in the plane the camera must face.",
    },
    "knee_angle": {
        "view": VIEW_SIDE_ON,
        "why": "The knee bends in the plane of travel; from behind, the shank is foreshortened.",
    },
    "hip_height": {
        "view": VIEW_SIDE_ON,
        "why": "Measured as a fraction of body height in the image, which needs an unforeshortened body.",
    },
    "contact_offset": {
        "view": VIEW_SIDE_ON,
        "why": "How far in front of the body the shuttle was struck is a depth measurement from behind.",
    },
    "stance_width": {
        "view": VIEW_REAR,
        "why": "Feet apart across the body is a depth measurement from the side.",
    },
    "landing_width": {
        "view": VIEW_REAR,
        "why": "Landing base is measured across the body, which a side view cannot see.",
    },
    "knee_over_foot": {
        "view": VIEW_REAR,
        "why": "Alignment of the knee over the foot is a frontal-plane observation.",
    },
}


def view_class(camera: CameraGeometry) -> dict:
    """Which way the camera looks at the court, and what that permits.

    The court has two axes and 2-D angles only survive when the camera faces
    one of them squarely. Since the camera's position is known in court metres,
    this is arithmetic rather than guesswork.

    Azimuth is measured from the court's long axis: 0 degrees is directly
    behind a baseline, 90 degrees is square to the side. Elevation is the angle
    down from horizontal to the middle of the court, which is what governs how
    fast the far-court depth error grows.

    Args:
        camera: Recovered camera geometry.

    Returns:
        A dict with ``view`` (one of :data:`VIEW_CLASSES`), ``azimuth_deg``,
        ``elevation_deg``, a one-line ``description``, and ``available`` /
        ``hidden`` lists naming the metric families this view does and does not
        support. ``hidden`` entries carry the reason and the fix.
    """
    from mikiri.court import CENTRE_X, NET_Y

    g = np.asarray(camera.ground_point, dtype=float)
    d = g - np.array([CENTRE_X, NET_Y])
    # 0 when the camera sits along the court's length, 90 when out to the side.
    azimuth = math.degrees(math.atan2(abs(float(d[0])), abs(float(d[1]))))
    horizontal = float(np.hypot(d[0], d[1]))
    elevation = math.degrees(math.atan2(camera.height_m, max(horizontal, 1e-6)))

    if azimuth <= VIEW_CONE_DEG:
        view = VIEW_REAR
        description = (
            "Behind the court: good for footwork and court position, poor for joint angles."
        )
    elif azimuth >= 90.0 - VIEW_CONE_DEG:
        view = VIEW_SIDE_ON
        description = "Side-on: good for joint angles, poor for position along the court."
    else:
        view = VIEW_OBLIQUE
        description = (
            "Oblique: square to neither court axis, so image-plane angles are "
            "foreshortened by an amount nothing can undo."
        )

    available, hidden = [], []
    for name, spec in VIEW_REQUIREMENTS.items():
        if spec["view"] == view:
            available.append(name)
        else:
            hidden.append(
                {
                    "metric": name,
                    "needs": spec["view"],
                    "why": spec["why"],
                    "fix": _view_fix(spec["view"]),
                }
            )
    return {
        "view": view,
        "azimuth_deg": round(azimuth, 1),
        "elevation_deg": round(elevation, 1),
        "description": description,
        "available": available,
        "hidden": hidden,
    }


def _view_fix(view: str) -> str:
    """Where to put the camera to unlock a view's measurements."""
    if view == VIEW_SIDE_ON:
        return "Film from the side of the court, level with the player, perpendicular to the net."
    if view == VIEW_REAR:
        return "Film from behind the baseline on the centre line, as high as you can get."
    return "Move the camera square to one of the court's two axes."
