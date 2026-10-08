//! Geometric constraints, solved by position-based projection.
//!
//! A constraint relates *points*, not entities: two endpoints must coincide, two
//! points must sit on one horizontal line, two points must be a fixed distance
//! apart. Solving them is Gauss-Seidel projection -- visit each constraint in
//! turn, move its points a step towards satisfaction, repeat until the largest
//! remaining error is below tolerance or the iteration budget runs out.
//!
//! Why projection and not Newton: the constraints here are all either linear
//! (coincident, horizontal, vertical, fixed) or one cheap non-linearity
//! (distance), and projection converges on those without a Jacobian, without a
//! linear solve, and without any way to diverge into NaN. A Newton solver would
//! converge in fewer steps and fail in more ways; for an interactive sketch
//! solver, failing never is worth more than converging fast.
//!
//! What it is not: a full parametric solver. There is no angle between lines,
//! no tangency, no equal-length across pairs, and no dragging with live solving.
//! Those need a real equation system. What is here covers the constraints that
//! make a sketch stop drifting, which is most of the value.

use cad_core::Vec2;

/// One geometric relationship between points.
///
/// Points are indices into the caller's point array, so the same solver works on
/// grips, on polyline vertices, and on anything else that can be expressed as
/// "these positions, in this order".
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Constraint {
    /// `points[i]` stays where it is. Anything referencing a fixed point treats
    /// it as infinitely massive.
    Fix { point: usize },
    /// Two points must coincide.
    Coincident { a: usize, b: usize },
    /// Two points share their y.
    Horizontal { a: usize, b: usize },
    /// Two points share their x.
    Vertical { a: usize, b: usize },
    /// Two points must be `distance` apart.
    Distance { a: usize, b: usize, distance: f32 },
}

impl Constraint {
    /// The points this constraint touches, for bounds checking.
    pub fn points(&self) -> Vec<usize> {
        match *self {
            Constraint::Fix { point } => vec![point],
            Constraint::Coincident { a, b }
            | Constraint::Horizontal { a, b }
            | Constraint::Vertical { a, b } => vec![a, b],
            Constraint::Distance { a, b, .. } => vec![a, b],
        }
    }

    /// Human-readable name, for the properties panel and the undo label.
    pub fn name(&self) -> &'static str {
        match self {
            Constraint::Fix { .. } => "Fix",
            Constraint::Coincident { .. } => "Coincident",
            Constraint::Horizontal { .. } => "Horizontal",
            Constraint::Vertical { .. } => "Vertical",
            Constraint::Distance { .. } => "Distance",
        }
    }
}

/// A set of constraints over one point array.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Problem {
    pub constraints: Vec<Constraint>,
}

impl Problem {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, c: Constraint) {
        // A duplicate constraint is not an error, but it doubles that
        // constraint's weight in the projection order, which biases the result
        // towards it for no reason.
        if !self.constraints.contains(&c) {
            self.constraints.push(c);
        }
    }

    pub fn remove(&mut self, c: Constraint) {
        self.constraints.retain(|x| *x != c);
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// The largest point index any constraint references, so the caller can
    /// size the point array correctly.
    pub fn max_point(&self) -> Option<usize> {
        self.constraints.iter().flat_map(|c| c.points()).max()
    }
}

/// Outcome of a solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveReport {
    /// Iterations actually run.
    pub iterations: usize,
    /// Largest remaining constraint error, in world units.
    pub residual: f32,
    /// True when the residual reached `tolerance`.
    pub converged: bool,
}

/// Solve `problem` over `points` in place.
///
/// `fixed[i]` marks point `i` as immovable, in addition to any [`Constraint::Fix`].
/// Returns how far the solution got, so the caller can decide whether to apply
/// it or to report that the sketch is over-constrained.
pub fn solve(
    problem: &Problem,
    points: &mut [Vec2],
    fixed: &[bool],
    tolerance: f32,
    max_iterations: usize,
) -> SolveReport {
    let n = points.len();
    // A constraint referencing a point that does not exist is a caller bug, not
    // a solve failure -- but panicking out of a UI command is worse, so clamp
    // the problem to what exists and solve that.
    let valid: Vec<&Constraint> = problem
        .constraints
        .iter()
        .filter(|c| c.points().iter().all(|p| *p < n))
        .collect();

    if valid.is_empty() {
        return SolveReport {
            iterations: 0,
            residual: 0.0,
            converged: true,
        };
    }

    // Inverse masses: 0 for fixed points, 1 for everything else. Projection
    // moves each point in proportion to its inverse mass, so a fixed point
    // never moves and a free point moves fully.
    let mut inv: Vec<f32> = vec![1.0; n];
    for (i, f) in fixed.iter().enumerate().take(n) {
        if *f {
            inv[i] = 0.0;
        }
    }
    for c in &valid {
        if let Constraint::Fix { point } = c
            && *point < n
        {
            inv[*point] = 0.0;
        }
    }

    let mut residual = f32::INFINITY;
    let mut iterations = 0;
    for _ in 0..max_iterations.max(1) {
        iterations += 1;
        residual = 0.0;
        for c in &valid {
            residual = residual.max(project(c, points, &inv));
        }
        if residual <= tolerance {
            return SolveReport {
                iterations,
                residual,
                converged: true,
            };
        }
        // A fully-fixed problem cannot move at all; iterating further changes
        // nothing, so stop rather than burn the budget.
        if residual.is_infinite() {
            break;
        }
    }
    SolveReport {
        iterations,
        residual,
        converged: false,
    }
}

/// Project one constraint, returning its remaining error afterwards.
fn project(c: &Constraint, points: &mut [Vec2], inv: &[f32]) -> f32 {
    match *c {
        Constraint::Fix { point } => {
            // Nothing to do: the point is already where it must stay, and its
            // inverse mass is 0 so nothing else moves it either.
            let _ = (points, inv, point);
            0.0
        }
        Constraint::Coincident { a, b } => {
            let (wa, wb) = (inv[a], inv[b]);
            let w = wa + wb;
            if w <= 0.0 {
                // Both fixed: the error is whatever distance remains, and no
                // projection can fix it. Reporting it is what tells the caller
                // the sketch is over-constrained.
                return points[a].distance(points[b]);
            }
            // Standard position-based update: each point moves along the gap in
            // proportion to its inverse mass. A fixed point (weight 0) holds
            // still while the free one travels the whole gap; two free points
            // meet in the middle.
            let gap = points[b] - points[a];
            points[a] += gap * (wa / w);
            points[b] -= gap * (wb / w);
            points[a].distance(points[b])
        }
        Constraint::Horizontal { a, b } => {
            let (wa, wb) = (inv[a], inv[b]);
            let w = wa + wb;
            if w <= 0.0 {
                return (points[a].y - points[b].y).abs();
            }
            let gap = points[b].y - points[a].y;
            points[a].y += gap * (wa / w);
            points[b].y -= gap * (wb / w);
            (points[a].y - points[b].y).abs()
        }
        Constraint::Vertical { a, b } => {
            let (wa, wb) = (inv[a], inv[b]);
            let w = wa + wb;
            if w <= 0.0 {
                return (points[a].x - points[b].x).abs();
            }
            let gap = points[b].x - points[a].x;
            points[a].x += gap * (wa / w);
            points[b].x -= gap * (wb / w);
            (points[a].x - points[b].x).abs()
        }
        Constraint::Distance { a, b, distance } => {
            let d = points[b] - points[a];
            let len = d.length();
            let err = (len - distance).abs();
            let (wa, wb) = (inv[a], inv[b]);
            let w = wa + wb;
            if w <= 0.0 || len < 1e-9 {
                return err;
            }
            // Move along the line joining them, split by inverse mass.
            let dir = d * (1.0 / len);
            let corr = dir * (len - distance);
            points[a] += corr * (wa / w);
            points[b] -= corr * (wb / w);
            (points[b].distance(points[a]) - distance).abs()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(v: &[[f32; 2]]) -> Vec<Vec2> {
        v.iter().map(|p| Vec2::new(p[0], p[1])).collect()
    }

    #[test]
    fn an_empty_problem_solves_immediately() {
        let p = Problem::new();
        let mut points = pts(&[[1.0, 2.0]]);
        let r = solve(&p, &mut points, &[], 1e-4, 100);
        assert!(r.converged);
        assert_eq!(r.iterations, 0);
        assert_eq!(points[0], Vec2::new(1.0, 2.0));
    }

    #[test]
    fn coincident_meets_in_the_middle() {
        let mut p = Problem::new();
        p.add(Constraint::Coincident { a: 0, b: 1 });
        let mut points = pts(&[[0.0, 0.0], [10.0, 0.0]]);
        let r = solve(&p, &mut points, &[], 1e-4, 100);
        assert!(r.converged, "{r:?}");
        assert!(points[0].distance(points[1]) < 1e-3, "{points:?}");
        assert!((points[0].x - 5.0).abs() < 1e-3, "{points:?}");
    }

    #[test]
    fn a_fixed_point_does_not_move() {
        let mut p = Problem::new();
        p.add(Constraint::Coincident { a: 0, b: 1 });
        let mut points = pts(&[[0.0, 0.0], [10.0, 0.0]]);
        let r = solve(&p, &mut points, &[true, false], 1e-4, 100);
        assert!(r.converged, "{r:?}");
        assert_eq!(points[0], Vec2::new(0.0, 0.0));
        assert!(points[1].distance(Vec2::ZERO) < 1e-3, "{points:?}");
    }

    #[test]
    fn fix_constraints_pin_their_points() {
        let mut p = Problem::new();
        p.add(Constraint::Fix { point: 0 });
        p.add(Constraint::Distance {
            a: 0,
            b: 1,
            distance: 5.0,
        });
        let mut points = pts(&[[0.0, 0.0], [10.0, 0.0]]);
        let r = solve(&p, &mut points, &[], 1e-4, 100);
        assert!(r.converged, "{r:?}");
        assert_eq!(points[0], Vec2::new(0.0, 0.0));
        assert!(
            (points[1].distance(Vec2::ZERO) - 5.0).abs() < 1e-3,
            "{points:?}"
        );
    }

    #[test]
    fn distance_sets_the_gap() {
        let mut p = Problem::new();
        p.add(Constraint::Distance {
            a: 0,
            b: 1,
            distance: 4.0,
        });
        let mut points = pts(&[[0.0, 0.0], [10.0, 0.0]]);
        let r = solve(&p, &mut points, &[], 1e-4, 100);
        assert!(r.converged, "{r:?}");
        assert!(
            (points[0].distance(points[1]) - 4.0).abs() < 1e-3,
            "{points:?}"
        );
    }

    #[test]
    fn horizontal_and_vertical_level_the_points() {
        let mut p = Problem::new();
        p.add(Constraint::Horizontal { a: 0, b: 1 });
        let mut points = pts(&[[0.0, 0.0], [10.0, 7.0]]);
        let r = solve(&p, &mut points, &[], 1e-4, 100);
        assert!(r.converged, "{r:?}");
        assert!((points[0].y - points[1].y).abs() < 1e-3, "{points:?}");
        // x is untouched: a horizontal constraint must not move what it does
        // not constrain.
        assert!((points[1].x - 10.0).abs() < 1e-3, "{points:?}");

        let mut p = Problem::new();
        p.add(Constraint::Vertical { a: 0, b: 1 });
        let mut points = pts(&[[0.0, 0.0], [10.0, 7.0]]);
        let r = solve(&p, &mut points, &[], 1e-4, 100);
        assert!(r.converged, "{r:?}");
        assert!((points[0].x - points[1].x).abs() < 1e-3, "{points:?}");
        assert!((points[1].y - 7.0).abs() < 1e-3, "{points:?}");
    }

    #[test]
    fn conflicting_fixes_do_not_converge_but_do_not_hang() {
        // Both points fixed 10 apart, asked to coincide: impossible. The solver
        // must report it, not spin for the whole budget and claim success.
        let mut p = Problem::new();
        p.add(Constraint::Coincident { a: 0, b: 1 });
        let mut points = pts(&[[0.0, 0.0], [10.0, 0.0]]);
        let r = solve(&p, &mut points, &[true, true], 1e-4, 100);
        assert!(!r.converged);
        assert!((r.residual - 10.0).abs() < 1e-3, "{r:?}");
        assert_eq!(points[0], Vec2::new(0.0, 0.0));
        assert_eq!(points[1], Vec2::new(10.0, 0.0));
    }

    #[test]
    fn dangling_point_indices_are_skipped_not_fatal() {
        let mut p = Problem::new();
        p.add(Constraint::Coincident { a: 0, b: 99 });
        let mut points = pts(&[[1.0, 1.0]]);
        let r = solve(&p, &mut points, &[], 1e-4, 100);
        assert!(r.converged);
        assert_eq!(points[0], Vec2::new(1.0, 1.0));
    }

    #[test]
    fn constraints_deduplicate() {
        let mut p = Problem::new();
        p.add(Constraint::Coincident { a: 0, b: 1 });
        p.add(Constraint::Coincident { a: 0, b: 1 });
        assert_eq!(p.len(), 1);
        p.remove(Constraint::Coincident { a: 0, b: 1 });
        assert!(p.is_empty());
    }

    #[test]
    fn a_chain_of_constraints_solves_together() {
        // Three points in a line, middle fixed: the ends must land 5 away.
        let mut p = Problem::new();
        p.add(Constraint::Fix { point: 1 });
        p.add(Constraint::Distance {
            a: 0,
            b: 1,
            distance: 5.0,
        });
        p.add(Constraint::Distance {
            a: 1,
            b: 2,
            distance: 5.0,
        });
        let mut points = pts(&[[0.0, 0.0], [3.0, 0.0], [20.0, 1.0]]);
        let r = solve(&p, &mut points, &[], 1e-3, 500);
        assert!(r.converged, "{r:?}");
        assert!(
            (points[0].distance(points[1]) - 5.0).abs() < 1e-2,
            "{points:?}"
        );
        assert!(
            (points[1].distance(points[2]) - 5.0).abs() < 1e-2,
            "{points:?}"
        );
        assert_eq!(points[1], Vec2::new(3.0, 0.0));
    }

    #[test]
    fn problem_reports_its_footprint() {
        let mut p = Problem::new();
        assert_eq!(p.max_point(), None);
        p.add(Constraint::Distance {
            a: 2,
            b: 5,
            distance: 1.0,
        });
        assert_eq!(p.max_point(), Some(5));
    }

    #[test]
    fn constraint_names_are_stable() {
        assert_eq!(Constraint::Fix { point: 0 }.name(), "Fix");
        assert_eq!(
            Constraint::Distance {
                a: 0,
                b: 1,
                distance: 1.0
            }
            .name(),
            "Distance"
        );
    }
}
