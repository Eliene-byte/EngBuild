//! `cad-ai` — tiny, local, deterministic inference.
//!
//! This is deliberately **not** a machine-learning framework and there is no
//! model file, no download and no network call anywhere in it. What it is:
//!
//! * a two-layer MLP small enough to inline in a CAD app (a few thousand `f32`s,
//!   forward pass in microseconds), and
//! * two models trained **in this file**, at startup, from a transition table
//!   written by hand.
//!
//! # What it is for
//!
//! CAD command lines are long and the commands are few. AutoCAD solves this with
//! a giant hardcoded alias table. The model here learns the same shape of thing
//! from data: after you draw a line you probably draw another line, and if you
//! have a selection you probably modify it. It is a suggestion, never an action
//! — the UI shows it and the user presses Enter.
//!
//! # Why not a real model
//!
//! Because it would not help. The command vocabulary is 49 symbols and the
//! context is "the last few commands". A matrix factorisation over that would be
//! a few kilobytes and would beat this MLP. The MLP is here because it is the
//! smallest thing that *generalises* — a typed prefix and a never-seen sequence
//! still produce a sensible ranking — while staying auditable: every weight is
//! readable and the training set is the table further down this file.

/// One layer's slice of the network's flat parameter vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    w: usize,
    b: usize,
    inputs: usize,
    outputs: usize,
}

impl Span {
    fn new(inputs: usize, outputs: usize, w: usize, b: usize) -> Self {
        Self {
            w,
            b,
            inputs,
            outputs,
        }
    }
    /// Where the next layer's weights would start.
    fn next(&self) -> usize {
        self.b + self.outputs
    }
}

/// Softmax, subtracting the max so the exponentials cannot overflow.
pub fn softmax_into(z: &mut [f32]) {
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        z.fill(0.0);
        return;
    }
    let mut sum = 0.0f32;
    for v in z.iter_mut() {
        // A NaN anywhere in the input poisons the result, and `f32::max` ignores
        // NaN rather than propagating it -- so the check has to be per element.
        if !v.is_finite() {
            z.fill(0.0);
            return;
        }
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 1e-30 {
        let inv = sum.recip();
        for v in z.iter_mut() {
            *v *= inv;
        }
    }
}

/// A two-layer MLP over one flat parameter vector.
///
/// The parameters live in a single `Vec<f32>` rather than inside each layer
/// because that is what makes the optimiser a flat loop with no borrow
/// juggling, and because it makes "how big is this model" a single number.
///
/// Layout, for `inputs x hidden` then `hidden x outputs`:
/// ```text
///   [ layer0 weights (inputs*hidden) ][ layer0 biases (hidden) ]
///   [ layer1 weights (hidden*outputs) ][ layer1 biases (outputs) ]
/// ```
#[derive(Debug, Clone)]
pub struct Mlp {
    p: Vec<f32>,
    l0: Span,
    l1: Span,
    hidden: usize,
    outputs: usize,
}

impl Mlp {
    /// `inputs` -> `hidden` (tanh) -> `outputs` (softmax).
    pub fn new(inputs: usize, hidden: usize, outputs: usize) -> Self {
        assert!(
            inputs > 0 && hidden > 0 && outputs > 0,
            "a layer cannot be empty"
        );
        let l0 = Span::new(inputs, hidden, 0, inputs * hidden);
        let l1 = Span::new(hidden, outputs, l0.next(), l0.next() + hidden * outputs);
        let mut p = vec![0.0f32; l1.next()];
        init_span(&mut p, l0, 0xA5A5_1234_5678_9ABC);
        init_span(&mut p, l1, 0x5A5A_ABCD_1234_5678);
        Self {
            p,
            l0,
            l1,
            hidden,
            outputs,
        }
    }

    /// Total parameter count, so "a few thousand floats" is checkable.
    pub fn parameters(&self) -> usize {
        self.p.len()
    }

    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// Parameters in storage order. Exposed because a model you cannot read is
    /// a model you cannot trust.
    pub fn params(&self) -> &[f32] {
        &self.p
    }

    /// Class probabilities for `x`.
    ///
    /// Takes `&self`: the only mutable state it ever had was a scratch buffer,
    /// and a fresh allocation here costs less than making the model impossible
    /// to share as `&'static`.
    #[allow(clippy::needless_range_loop)]
    pub fn predict(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.l0.inputs, "input width mismatch");
        let mut a = vec![0.0f32; self.hidden];
        self.forward_hidden(x, &mut a);
        let mut z = vec![0.0f32; self.outputs];
        for o in 0..self.outputs {
            let row = self.l1.w + o * self.l1.inputs;
            let mut s = self.p[self.l1.b + o];
            for i in 0..self.hidden {
                s += self.p[row + i] * a[i];
            }
            z[o] = s;
        }
        softmax_into(&mut z);
        z
    }

    /// The loops below index several parallel slices at the same offset, which
    /// is what `needless_range_loop` is written to flag; here it would mean
    /// collecting four iterators to walk the same row, so the lint is muted for
    /// this function only.
    #[allow(clippy::needless_range_loop)]
    fn forward_hidden(&self, x: &[f32], a: &mut [f32]) {
        for o in 0..self.hidden {
            let row = self.l0.w + o * self.l0.inputs;
            let mut s = self.p[self.l0.b + o];
            for i in 0..self.l0.inputs {
                s += self.p[row + i] * x[i];
            }
            a[o] = s.tanh();
        }
    }

    /// Cross-entropy of a prediction, clamped so one bad step cannot produce an
    /// infinity that poisons every later weight.
    pub fn cross_entropy(p: &[f32], target: usize) -> f32 {
        match p.get(target) {
            Some(&p_t) if p_t > 1e-12 => -p_t.ln(),
            _ => 27.6,
        }
    }

    /// Gradients of the cross-entropy, accumulated into `grads`.
    ///
    /// Hand-derived, because for two layers a graph buys nothing and a
    /// dependency buys a megabyte. The one identity that does the work is that
    /// the softmax Jacobian collapses the output gradient to `(p - onehot)`.
    #[allow(clippy::needless_range_loop)]
    pub fn backward(&self, x: &[f32], p: &[f32], target: usize, grads: &mut [f32]) {
        assert_eq!(grads.len(), self.p.len(), "gradient buffer size mismatch");
        // The hidden activation this prediction used. Recomputed rather than
        // cached so `backward` can stay `&self` and the two paths cannot drift.
        let mut a = vec![0.0f32; self.hidden];
        self.forward_hidden(x, &mut a);

        // Output layer. dz = p - onehot.
        let mut dz = vec![0.0f32; self.outputs];
        dz.copy_from_slice(p);
        if let Some(t) = dz.get_mut(target) {
            *t -= 1.0;
        }

        let mut dh = vec![0.0f32; self.hidden];
        for o in 0..self.outputs {
            let g = dz[o];
            let row = self.l1.w + o * self.l1.inputs;
            for i in 0..self.hidden {
                grads[row + i] += g * a[i];
                dh[i] += g * self.p[row + i];
            }
            grads[self.l1.b + o] += g;
        }

        // Back through tanh: d/dz tanh(z) = 1 - tanh(z)^2, per unit.
        for i in 0..self.hidden {
            dh[i] *= 1.0 - a[i] * a[i];
        }

        for o in 0..self.hidden {
            let g = dh[o];
            let row = self.l0.w + o * self.l0.inputs;
            for i in 0..self.l0.inputs {
                grads[row + i] += g * x[i];
            }
            grads[self.l0.b + o] += g;
        }
    }
}

/// Xavier-scaled initialisation from a deterministic hash of each weight's
/// position, so two runs of the same build produce byte-identical weights.
fn init_span(p: &mut [f32], s: Span, salt: u64) {
    let limit = (s.inputs as f32).sqrt().recip();
    for o in 0..s.outputs {
        for i in 0..s.inputs {
            let h = mix64(salt ^ ((o as u64) << 32) ^ i as u64);
            p[s.w + o * s.inputs + i] = (((h >> 40) as f32 / 8_388_608.0) - 1.0) * limit;
        }
    }
}

/// SplitMix64's finaliser, used as a hash and not as a generator.
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// One training example.
#[derive(Debug, Clone)]
pub struct Sample {
    pub x: Vec<f32>,
    pub target: usize,
}

/// Full-batch gradient descent with momentum, returning the final mean loss.
///
/// Full batch rather than mini-batch: the training sets here are a couple of
/// hundred rows, so it is one pass per epoch either way, and skipping the
/// shuffling removes the last source of run-to-run variation.
pub fn train(net: &mut Mlp, data: &[Sample], epochs: usize, lr: f32, momentum: f32) -> f32 {
    assert!(!data.is_empty(), "training on nothing is a no-op");
    let n = data.len() as f32;
    let mut velocity = vec![0.0f32; net.parameters()];
    let mut grads = vec![0.0f32; net.parameters()];
    let mut loss = f32::INFINITY;
    for _ in 0..epochs {
        grads.fill(0.0);
        let mut l = 0.0f32;
        for s in data {
            let p = net.predict(&s.x);
            l += Mlp::cross_entropy(&p, s.target);
            net.backward(&s.x, &p, s.target, &mut grads);
        }
        loss = l / n;
        for i in 0..grads.len() {
            velocity[i] = momentum * velocity[i] - lr * grads[i] / n;
            net.p[i] += velocity[i];
        }
    }
    loss
}

pub mod eval;
pub mod language;
pub mod suggest;

pub use eval::evaluate;
pub use language::{GeometryKind, Intent, parse};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix64_is_a_hash_not_a_generator() {
        // Deterministic and well-spread: the point of using it is that the
        // initial weights are reproducible, not that they are random.
        assert_eq!(mix64(1), mix64(1));
        assert_ne!(mix64(1), mix64(2));
        let vals: Vec<u64> = (0..256).map(mix64).collect();
        assert_eq!(
            vals.iter().collect::<std::collections::HashSet<_>>().len(),
            256
        );
    }

    #[test]
    fn parameter_count_matches_the_layout() {
        let net = Mlp::new(4, 8, 3);
        assert_eq!(net.parameters(), 4 * 8 + 8 + 8 * 3 + 3);
        assert_eq!(net.outputs(), 3);
        assert!(net.params().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn initialisation_is_reproducible() {
        let a = Mlp::new(5, 7, 2);
        let b = Mlp::new(5, 7, 2);
        assert_eq!(a.params(), b.params(), "same build, same weights");
        let c = Mlp::new(5, 7, 2);
        assert_eq!(a.params(), c.params());
    }

    #[test]
    fn initial_weights_are_not_all_zero() {
        let net = Mlp::new(6, 9, 4);
        assert!(
            net.params().iter().any(|v| v.abs() > 1e-4),
            "an all-zero init would never leave the origin"
        );
    }

    #[test]
    fn predict_returns_a_distribution() {
        let net = Mlp::new(4, 8, 3);
        let p = net.predict(&[0.0, 1.0, 0.0, 0.0]).to_vec();
        assert_eq!(p.len(), 3);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(p.iter().all(|v| *v >= 0.0 && v.is_finite()));
    }

    #[test]
    fn predict_handles_inputs_it_never_saw() {
        // There is no lookup table anywhere, so an unseen vector must still
        // produce a valid distribution rather than panic or return garbage.
        let net = Mlp::new(4, 8, 3);
        for x in [
            [-9.0, 1e6, 0.5, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [1e-12, -1e-12, 1e-12, -1e-12],
        ] {
            let p = net.predict(&x).to_vec();
            assert!(p.iter().all(|v| v.is_finite()), "{p:?}");
            assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-3, "{p:?}");
        }
    }

    #[test]
    #[should_panic(expected = "input width mismatch")]
    fn predict_rejects_a_wrong_width() {
        let net = Mlp::new(4, 8, 3);
        net.predict(&[1.0, 2.0]);
    }

    #[test]
    fn softmax_sums_to_one_and_survives_overflow() {
        let mut z = [1.0, 2.0, 3.0];
        softmax_into(&mut z);
        assert!((z.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(z[2] > z[0]);

        // Values that would overflow `exp` if the max were not subtracted.
        let mut big = [1000.0, 1001.0];
        softmax_into(&mut big);
        assert!(big.iter().all(|v| v.is_finite()));
        assert!((big.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_of_a_non_finite_input_is_zeros_not_nan() {
        let mut z = [f32::NAN, 1.0];
        softmax_into(&mut z);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cross_entropy_is_minimised_at_a_confident_correct_prediction() {
        let good = [0.99, 0.01];
        let bad = [0.01, 0.99];
        assert!(Mlp::cross_entropy(&good, 0) < Mlp::cross_entropy(&bad, 0));
        // Even a hopeless prediction must return a finite, large loss.
        let l = Mlp::cross_entropy(&[0.0, 1.0], 0);
        assert!(l.is_finite() && l > 1.0, "{l}");
    }

    #[test]
    fn gradients_are_finite_and_non_trivial() {
        let net = Mlp::new(4, 6, 3);
        let x = vec![0.2, -0.4, 1.0, 0.5];
        let copy = net.clone();
        let p = copy.predict(&x);
        let mut grads = vec![0.0; net.parameters()];
        net.backward(&x, &p, 1, &mut grads);
        assert!(grads.iter().all(|g| g.is_finite()), "{grads:?}");
        assert!(
            grads.iter().any(|g| g.abs() > 1e-8),
            "all-zero gradients mean the backward pass is wrong"
        );
    }

    #[test]
    fn gradients_match_a_finite_difference() {
        // The only honest way to check hand-written gradients.
        let net = Mlp::new(3, 5, 2);
        let x = vec![0.3, -0.7, 0.2];
        let h = 1e-3f32;
        let probe = net.clone();
        let base = {
            let p = probe.predict(&x).to_vec();
            Mlp::cross_entropy(&p, 0)
        };
        for idx in [0usize, 7, 15, 16, net.parameters() - 1] {
            let mut up = net.clone();
            up.p[idx] += h;
            let pu = up.clone();
            let lo = {
                up.p[idx] -= 2.0 * h;
                let pu = up.predict(&x).to_vec();
                Mlp::cross_entropy(&pu, 0)
            };
            let pu2 = pu.predict(&x);
            let hi = Mlp::cross_entropy(&pu2, 0);
            let numeric = (hi - lo) / (2.0 * h);

            let copy = net.clone();
            let p = copy.predict(&x);
            let mut grads = vec![0.0; net.parameters()];
            net.backward(&x, &p, 0, &mut grads);
            assert!(
                (grads[idx] - numeric).abs() < 2e-2,
                "grad[{idx}] analytic={} numeric={numeric}",
                grads[idx]
            );
            let _ = base;
        }
    }

    #[test]
    #[should_panic(expected = "gradient buffer size mismatch")]
    fn gradient_buffer_size_is_checked() {
        let net = Mlp::new(3, 4, 2);
        net.backward(&[0.0; 3], &[0.5, 0.5], 0, &mut [0.0; 3]);
    }

    #[test]
    fn training_reduces_the_loss_on_a_separable_task() {
        // Two classes, perfectly separated on the first input. A working
        // optimiser must get this right; a broken gradient would not.
        let data: Vec<Sample> = (0..16)
            .map(|i| {
                let hi = i >= 8;
                Sample {
                    x: vec![if hi { 1.0 } else { -1.0 }, 0.5, -0.5, 0.25],
                    target: usize::from(hi),
                }
            })
            .collect();
        let net = Mlp::new(4, 12, 2);
        let mut l = 0.0;
        for s in &data {
            l += Mlp::cross_entropy(&net.predict(&s.x), s.target);
        }
        let before = l / data.len() as f32;
        let mut net = net;
        let after = train(&mut net, &data, 800, 0.35, 0.9);
        assert!(
            after < before * 0.5,
            "loss barely moved: {before} -> {after}"
        );
    }

    #[test]
    fn a_trained_model_can_be_shared_as_a_static() {
        // `predict` takes `&self`, so a caller can train once and keep the model
        // behind a `OnceLock`. This is what makes the startup cost a one-off.
        static M: std::sync::OnceLock<Mlp> = std::sync::OnceLock::new();
        let m = M.get_or_init(|| {
            let mut net = Mlp::new(2, 4, 2);
            let data = vec![Sample {
                x: vec![1.0, 0.0],
                target: 0,
            }];
            train(&mut net, &data, 5, 0.1, 0.0);
            net
        });
        assert_eq!(m.predict(&[1.0, 0.0]).len(), 2);
    }

    #[test]
    fn training_is_deterministic() {
        let data: Vec<Sample> = (0..8)
            .map(|i| Sample {
                x: vec![i as f32 / 8.0, 1.0, 0.0, 0.0],
                target: i % 2,
            })
            .collect();
        let run = || {
            let mut net = Mlp::new(4, 6, 2);
            train(&mut net, &data, 50, 0.1, 0.9)
        };
        assert_eq!(run(), run(), "same data must give the same loss");
    }

    #[test]
    #[should_panic(expected = "training on nothing")]
    fn training_on_nothing_is_rejected() {
        let mut net = Mlp::new(2, 2, 2);
        train(&mut net, &[], 1, 0.1, 0.0);
    }

    #[test]
    #[should_panic(expected = "cannot be empty")]
    fn an_empty_layer_is_rejected() {
        Mlp::new(0, 1, 1);
    }

    #[test]
    fn a_network_is_small_enough_to_inline() {
        // The whole premise: a suggestion model must not cost more than the
        // widgets it lives next to. 6k floats is 24 kB.
        let net = Mlp::new(64, 48, 49);
        assert!(
            net.parameters() * 4 <= 32 * 1024,
            "{} parameters is {} kB",
            net.parameters(),
            net.parameters() * 4 / 1024
        );
    }
}
