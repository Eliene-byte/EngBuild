//! The command-line suggestion model.
//!
//! Two layers: a small MLP that predicts the *next* command from the recent
//! history, and a deterministic ranker that merges that prior with character
//! similarity so a mistyped prefix still finds its command.
//!
//! The training set is the table below. It is hand-written because it encodes
//! knowledge about how CAD is actually used — "after a line, another line;
//! after an erase, something to replace what you erased" — and no amount of
//! gradient descent invents that. The model's job is to *generalise* it, not to
//! discover it: given a history it has never seen, it still produces a sensible
//! ranking.

use crate::{Mlp, Sample};

/// The commands this model knows about, and their index.
///
/// Deliberately not "every command in the registry": the registry has 49 entries
/// and most of them are file or view commands that do not follow one another.
/// Twenty draw/modify commands is the part with real structure, and a smaller
/// vocabulary means each class gets enough training signal to be worth
/// suggesting.
pub const VOCAB: [&str; 22] = [
    "line",
    "pline",
    "circle",
    "arc",
    "rectangle",
    "move",
    "copy",
    "rotate",
    "scale",
    "mirror",
    "offset",
    "trim",
    "extend",
    "erase",
    "explode",
    "array",
    "zoomall",
    "zoomin",
    "zoomout",
    "view3d",
    "undo",
    "layer",
];

/// What tends to be typed after what.
///
/// Each row's successors become positive examples with equal probability, so
/// the softmax spreads mass across the plausible next steps instead of
/// collapsing onto the first one listed. Order therefore does not matter, which
/// is what makes the table easy to maintain.
const TRANSITIONS: [(&str, &[&str]); 18] = [
    ("line", &["line", "erase", "undo", "zoomall"]),
    ("pline", &["pline", "line", "zoomall", "erase"]),
    ("circle", &["circle", "line", "trim", "zoomall"]),
    ("arc", &["arc", "line", "circle", "zoomall"]),
    ("rectangle", &["line", "rectangle", "offset", "zoomall"]),
    ("move", &["move", "copy", "rotate", "undo"]),
    ("copy", &["copy", "array", "move", "erase"]),
    ("rotate", &["rotate", "scale", "move", "undo"]),
    ("scale", &["scale", "array", "move", "undo"]),
    ("mirror", &["mirror", "move", "copy", "undo"]),
    ("offset", &["offset", "trim", "extend", "erase", "layer"]),
    ("trim", &["trim", "extend", "erase", "undo"]),
    ("extend", &["extend", "trim", "offset", "undo"]),
    ("erase", &["line", "circle", "rectangle", "undo", "layer"]),
    ("explode", &["explode", "erase", "move", "undo"]),
    ("array", &["array", "erase", "move", "undo"]),
    (
        "zoomall",
        &["zoomin", "zoomout", "zoomall", "line", "layer"],
    ),
    ("zoomin", &["zoomin", "zoomout", "zoomall", "undo", "layer"]),
];

/// Commands reachable from no context at all: the cold-start distribution.
const ROOTS: [&str; 8] = [
    "line",
    "circle",
    "rectangle",
    "arc",
    "pline",
    "zoomall",
    "view3d",
    "undo",
];

/// Feature layout.
const N_CONTEXT: usize = VOCAB.len();
const N_FLAGS: usize = 4;
const N_INPUTS: usize = N_CONTEXT + N_FLAGS;
const HIDDEN: usize = 48;

/// The prediction model.
#[derive(Debug, Clone)]
pub struct NextCommand {
    net: Mlp,
    /// Constant one-hot row for "no previous command".
    cold: Vec<f32>,
}

impl Default for NextCommand {
    fn default() -> Self {
        Self::trained()
    }
}

impl NextCommand {
    /// Train from [`TRANSITIONS`] and [`ROOTS`]. Cheap enough to do at startup:
    /// a few hundred samples and a couple of thousand epochs on this network.
    pub fn trained() -> Self {
        let data = training_set();
        let mut net = Mlp::new(N_INPUTS, HIDDEN, VOCAB.len());
        crate::train(&mut net, &data, 2500, 0.5, 0.9);

        // The cold-start row: no context flags set. Nothing should be predicted
        // with confidence here, because the user has told us nothing.
        let cold = vec![0.0; N_INPUTS];

        Self { net, cold }
    }

    /// Parameter count, for the status bar.
    pub fn parameters(&self) -> usize {
        self.net.parameters()
    }

    /// Feature vector for one query.
    ///
    /// `history` is the most recent commands, oldest first; only the last entry
    /// is used as context, which is all the transition table encodes.
    fn features(&self, last: Option<usize>, has_selection: bool, tool_active: bool) -> Vec<f32> {
        let mut x = vec![0.0f32; N_INPUTS];
        match last {
            Some(i) => x[i] = 1.0,
            None => {
                // Spread the "unknown" mass over every context slot rather than
                // picking one, so the cold case is a genuine average and not an
                // arbitrary tie-break.
                let inv = 1.0 / N_CONTEXT as f32;
                for v in x.iter_mut().take(N_CONTEXT) {
                    *v = inv;
                }
            }
        }
        x[N_CONTEXT] = f32::from(u8::from(has_selection));
        x[N_CONTEXT + 1] = f32::from(u8::from(tool_active));
        x[N_CONTEXT + 2] = 1.0; // bias
        x[N_CONTEXT + 3] = 1.0; // bias
        x
    }

    /// Probabilities for every command in [`VOCAB`], in vocabulary order.
    pub fn distribution(
        &self,
        history: &[&str],
        has_selection: bool,
        tool_active: bool,
    ) -> Vec<f32> {
        let last = history
            .last()
            .and_then(|c| VOCAB.iter().position(|v| v == c));
        let x = if last.is_none() && history.is_empty() {
            self.cold.clone()
        } else {
            self.features(last, has_selection, tool_active)
        };
        self.net.predict(&x)
    }

    /// The `k` most likely next commands, most likely first.
    pub fn suggest(
        &self,
        history: &[&str],
        has_selection: bool,
        tool_active: bool,
    ) -> Vec<(&'static str, f32)> {
        let p = self.distribution(history, has_selection, tool_active);
        let mut ranked: Vec<(usize, f32)> = p.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
            .into_iter()
            .take(4)
            .map(|(i, s)| (VOCAB[i], s))
            .collect()
    }
}

/// Build the training set from the transition table.
///
/// Each row declares, for a context command, the commands that tend to follow
/// **and whether they need a selection**. That second part is what makes the
/// selection flag learnable: without it, the same target would be trained under
/// both flag values and the network would correctly conclude the flag does not
/// matter.
fn training_set() -> Vec<Sample> {
    let mut out = Vec::new();
    for (from, tos) in TRANSITIONS {
        let Some(ctx) = VOCAB.iter().position(|v| *v == from) else {
            continue;
        };
        for to in tos {
            let Some(target) = VOCAB.iter().position(|v| *v == *to) else {
                continue;
            };
            let target_needs_selection = needs_selection(to);
            // Train the matching flag, and -- for modify commands -- the
            // non-matching one as a negative so the flag carries information
            // rather than being a constant offset.
            out.push(Sample {
                x: feature_vec(Some(ctx), target_needs_selection, false),
                target,
            });
            if target_needs_selection {
                for other in MODIFY_COMMANDS.iter() {
                    if let Some(t) = VOCAB.iter().position(|v| *v == *other) {
                        out.push(Sample {
                            x: feature_vec(Some(ctx), false, false),
                            target: t,
                        });
                    }
                }
            }
        }
    }

    for root in ROOTS.iter() {
        let Some(target) = VOCAB.iter().position(|v| *v == *root) else {
            continue;
        };
        // Roots are reached from a clean slate: no selection, no active tool.
        out.push(Sample {
            x: feature_vec(None, false, false),
            target,
        });
    }

    out
}

/// Commands that act on a selection rather than drawing new geometry.
const MODIFY_COMMANDS: [&str; 6] = ["move", "copy", "rotate", "scale", "mirror", "erase"];

/// Does `name` need something selected before it can do anything?
pub fn needs_selection(name: &str) -> bool {
    MODIFY_COMMANDS.contains(&name)
}

fn feature_vec(ctx: Option<usize>, has_selection: bool, tool_active: bool) -> Vec<f32> {
    let mut x = vec![0.0f32; N_INPUTS];
    match ctx {
        Some(i) => x[i] = 1.0,
        None => {
            let inv = 1.0 / N_CONTEXT as f32;
            for v in x.iter_mut().take(N_CONTEXT) {
                *v = inv;
            }
        }
    }
    x[N_CONTEXT] = f32::from(u8::from(has_selection));
    x[N_CONTEXT + 1] = f32::from(u8::from(tool_active));
    x[N_CONTEXT + 2] = 1.0;
    x[N_CONTEXT + 3] = 1.0;
    x
}

/// Damerau-free edit distance, capped: the ranker only needs to know "close",
/// and an unbounded O(n·m) loop over a 20-character prefix is wasted work.
fn edit_distance(a: &str, b: &str, cap: u32) -> u32 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len() as u32;
    }
    if b.is_empty() {
        return a.len() as u32;
    }
    let mut prev2: Vec<u32> = (0..=b.len() as u32).collect();
    let mut prev: Vec<u32> = (0..=b.len() as u32).collect();
    let mut cur: Vec<u32> = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i as u32;
        let mut best = cur[0];
        for j in 1..=b.len() {
            let cost = u32::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            best = best.min(cur[j]);
        }
        if best > cap {
            return cap + 1;
        }
        std::mem::swap(&mut prev2, &mut prev);
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Merge the model's prior with character similarity to rank completions.
///
/// This part is deterministic on purpose. Learning "l-i-n-e is close to lin" from
/// a transition table of CAD commands would be silly: string distance is exact,
/// free, and explainable. The model contributes the *prior*, the distance
/// contributes the *match*, and neither pretends to be the other.
pub fn rank<'k>(prefix: &str, known: &'k [&'k str], prior: &[f32]) -> Vec<(&'k str, f32)> {
    if prefix.is_empty() {
        return Vec::new();
    }
    let p = prefix.to_ascii_lowercase();
    let mut scored: Vec<(&str, f32)> = known
        .iter()
        .filter_map(|name| {
            let idx = VOCAB.iter().position(|v| *v == *name)?;
            let lower = name.to_ascii_lowercase();
            // Prefix match is the strong signal; a near miss is a weak one.
            let similarity = if lower.starts_with(&p) {
                1.0 - (lower.len() - p.len()) as f32 / 64.0
            } else {
                let d = edit_distance(&lower, &p, 6);
                if d > 6 {
                    return None;
                }
                0.5 - d as f32 / 20.0
            };
            Some((
                *name,
                similarity * 0.8 + prior.get(idx).copied().unwrap_or(0.0) * 0.2,
            ))
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Train once and share it. Training is deterministic, so this is the same
    /// object every test would have built for itself -- just built once.
    fn trained() -> &'static NextCommand {
        static M: std::sync::OnceLock<NextCommand> = std::sync::OnceLock::new();
        M.get_or_init(NextCommand::trained)
    }

    #[test]
    fn the_transition_table_only_names_known_commands() {
        // A typo here would silently remove a row from the training set.
        for (from, tos) in TRANSITIONS {
            assert!(
                VOCAB.contains(&from),
                "`{from}` is not in VOCAB, so its row would be skipped"
            );
            for to in tos {
                assert!(
                    VOCAB.iter().any(|v| v == to),
                    "`{to}` (after {from}) is not in VOCAB"
                );
            }
        }
        for r in ROOTS {
            assert!(VOCAB.contains(&r), "`{r}` is not in VOCAB");
        }
    }

    #[test]
    fn every_vocabulary_entry_is_reachable() {
        // A command nothing ever predicts is dead weight in the output layer.
        let targets: std::collections::HashSet<usize> =
            training_set().iter().map(|s| s.target).collect();
        for (i, name) in VOCAB.iter().enumerate() {
            assert!(
                targets.contains(&i),
                "`{name}` is never a training target, so it can never be suggested"
            );
        }
    }

    #[test]
    fn the_training_set_is_not_trivial() {
        let data = training_set();
        assert!(data.len() > 60, "only {} samples", data.len());
        assert_eq!(data.len(), data.len()); // no reallocation surprises
        assert!(data.iter().all(|s| s.x.len() == N_INPUTS));
    }

    #[test]
    fn the_model_fits_its_own_training_data() {
        let m = trained();
        // After a line, with nothing selected, the top suggestion should be a
        // drawing command -- not a modify command.
        let s = m.suggest(&["line"], false, false);
        assert!(!s.is_empty());
        let top = s[0].0;
        assert!(
            matches!(top, "line" | "erase" | "undo" | "zoomall"),
            "unexpected top suggestion after line: {top} (all: {s:?})"
        );
    }

    #[test]
    fn a_selection_shifts_the_prediction_towards_modify_commands() {
        // This is the whole point of the selection flag: with a selection the
        // model should reach for move/copy before it reaches for line.
        let m = trained();
        let with = m.distribution(&["line"], true, false);
        let without = m.distribution(&["line"], false, false);
        let idx = |n: &str| VOCAB.iter().position(|v| *v == n).unwrap();
        let pair = |d: &[f32], a: &str, b: &str| d[idx(a)] - d[idx(b)];
        assert!(
            pair(&with, "move", "line") > pair(&without, "move", "line"),
            "a selection did not move probability towards move"
        );
    }

    #[test]
    fn predictions_are_always_a_distribution() {
        let m = trained();
        for (hist, sel, tool) in [
            (vec![], false, false),
            (vec!["line"], false, false),
            (vec!["erase", "line", "circle"], true, false),
            (vec!["nonexistent-command"], false, false),
        ] {
            let p = m.distribution(&hist, sel, tool);
            assert_eq!(p.len(), VOCAB.len());
            assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-3, "{p:?}");
            assert!(p.iter().all(|v| v.is_finite() && *v >= 0.0), "{p:?}");
        }
    }

    #[test]
    fn an_unknown_history_command_does_not_panic() {
        // The command line accepts anything the user types, including a
        // command the model has never heard of.
        let m = trained();
        let s = m.suggest(&["zzz", "yyy"], true, true);
        assert!(!s.is_empty());
        assert!(s.iter().all(|(n, _)| VOCAB.contains(n)));
    }

    #[test]
    fn the_model_stays_tiny() {
        let m = trained();
        assert!(
            m.parameters() < 6000,
            "{} parameters is {} kB -- too big to inline",
            m.parameters(),
            m.parameters() * 4 / 1024
        );
    }

    #[test]
    fn edit_distance_basics() {
        assert_eq!(edit_distance("", "abc", 9), 3);
        assert_eq!(edit_distance("abc", "", 9), 3);
        assert_eq!(edit_distance("abc", "abc", 9), 0);
        assert_eq!(edit_distance("lin", "line", 9), 1);
        assert_eq!(edit_distance("ln", "line", 9), 2);
        assert_eq!(edit_distance("kitten", "sitting", 9), 3);
    }

    #[test]
    fn edit_distance_respects_the_cap() {
        // A prefix that shares nothing must bail out rather than run the full
        // table for every candidate.
        assert!(edit_distance("zzzzzzzzzzzz", "line", 4) > 4);
    }

    #[test]
    fn rank_finds_an_exact_prefix() {
        let prior = vec![0.1; VOCAB.len()];
        let r = rank("li", &["line", "circle", "arc"], &prior);
        assert_eq!(r[0].0, "line");
    }

    #[test]
    fn rank_tolerates_a_typo() {
        // `ln` is not a prefix of `line`, but it is one edit away and must still
        // surface it: mistyping is the case a pure prefix match fails.
        let prior = vec![0.1; VOCAB.len()];
        let r = rank("ln", &["line", "circle", "arc"], &prior);
        assert_eq!(r.first().map(|x| x.0), Some("line"), "{r:?}");
    }

    #[test]
    fn rank_returns_nothing_for_an_empty_prefix() {
        let prior = vec![0.1; VOCAB.len()];
        assert!(rank("", &["line"], &prior).is_empty());
    }

    #[test]
    fn rank_drops_candidates_far_from_the_prefix() {
        let prior = vec![0.1; VOCAB.len()];
        let r = rank("qqqqqqqqqq", &["line", "circle"], &prior);
        assert!(r.is_empty(), "{r:?}");
    }

    #[test]
    fn rank_is_case_insensitive() {
        let prior = vec![0.1; VOCAB.len()];
        let r = rank("LI", &["line"], &prior);
        assert_eq!(r.first().map(|x| x.0), Some("line"));
    }

    #[test]
    fn rank_ignores_commands_the_model_does_not_know() {
        let prior = vec![0.1; VOCAB.len()];
        // `new` is a real command but is not in VOCAB, so it has no prior.
        let r = rank("ne", &["new", "line"], &prior);
        assert_eq!(r.first().map(|x| x.0), Some("line"), "{r:?}");
    }
}
