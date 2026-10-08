//! `cad-ai` — tiny, local, deterministic inference.
//!
//! This crate is deliberately **not** a machine-learning framework and there is no
//! model file, no download and no network call anywhere in it. What it holds:
//!
//! * a two-layer MLP whose hand-written gradients are checked against finite
//!   differences;
//! * a command-line suggestion model, trained in code from a hand-written
//!   transition table;
//! * a natural-language intent parser, which is where "superior to AutoCAD in
//!   how it uses AI" actually lives.
//!
//! # Why natural language
//!
//! AutoCAD's command line is forty years of aliasing: `L`, `LINE`, `_LINE`,
//! `-LINE` and a hundred others all mean the same thing, and the user has to know
//! which. An intent parser means the user can say what they want instead. It
//! is also cheap: CAD commands are a small vocabulary with a strict shape, so a
//! keyword-and-number parser generalises better here than any learned model
//! would, and it is explainable -- when it misunderstands, the words it got are
//! right there in the input.
//!
//! # What is in scope
//!
//! Parsing "draw a line from 0,0 to 10,10" and "circle at 5,5 radius 3". Out of
//! scope: free-form conversational commands ("make it look nicer"). That needs a
//! model that is not a few hundred lines, and pretending otherwise is how
//! products ship a feature that fails in the field.

use crate::suggest::VOCAB;

/// A number with a unit, as the user typed it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity {
    pub value: f32,
    pub suffix: &'static str,
}

/// What the user asked for.
#[derive(Debug, Clone, PartialEq)]
pub enum Intent {
    /// Run a built-in command with its argument string.
    ///
    /// `name` is owned rather than borrowed: it comes from the user's input, which
    /// is a local `String` in `parse`, and a parse happens once per submitted
    /// line, so the allocation is not on a hot path.
    Command { name: String, args: String },
    /// A drawing command with resolved geometry, ready to run.
    Geometry {
        kind: GeometryKind,
        a: [f32; 2],
        b: [f32; 2],
    },
    /// Nothing recognised. The caller decides whether to ask or to guess.
    Unknown,
}

/// Geometry the parser can resolve without picking anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryKind {
    Line,
    Circle,
    Rectangle,
}

impl GeometryKind {
    /// The command this geometry maps onto.
    pub fn command(self) -> &'static str {
        match self {
            GeometryKind::Line => "line",
            GeometryKind::Circle => "circle",
            GeometryKind::Rectangle => "rectangle",
        }
    }

    /// The natural-language noun, for "did you mean a circle?".
    pub fn noun(self) -> &'static str {
        match self {
            GeometryKind::Line => "line",
            GeometryKind::Circle => "circle",
            GeometryKind::Rectangle => "rectangle",
        }
    }
}

/// Parse a natural-language request into something runnable.
///
/// Returns [`Intent::Unknown`] when nothing recognises it, which is a real answer
/// and not a failure: guessing at a command the user did not ask for would put
/// geometry in the drawing by mistake.
///
/// ```text
///   "draw a line from 0,0 to 10,10"      -> Geometry::Line (0,0)-(10,10)
///   "circle at 5,5 radius 3"             -> Geometry::Circle
///   "line 0 0 10 10"                     -> Geometry::Line
///   "zoom all"                           -> Command("zoomall")
///   "erase the thing I just drew"         -> Command("erase")
/// ```
pub fn parse(text: &str) -> Intent {
    let t = normalize(text);
    if t.is_empty() {
        return Intent::Unknown;
    }

    // A leading "=" is arithmetic, not a command: the command line doubles as a
    // calculator, and that has to win over the command parser or every formula
    // would be a "command not found".
    if let Some(expr) = t.strip_prefix('=').map(str::trim) {
        if let Some(v) = crate::eval::evaluate(expr) {
            return Intent::Command {
                name: "print".to_string(),
                args: format_number(v),
            };
        }
        return Intent::Unknown;
    }

    // Geometry first: it is the only branch that produces entities, and a
    // sentence naming a shape is not a command name no matter what else it says.
    //
    // A shape keyword that is *present* but incomplete is Unknown, not a command.
    // Falling through would run `circle` and leave the user's centre on the floor.
    if names_a_shape(&t) {
        return geometry(&t).unwrap_or(Intent::Unknown);
    }

    // Then a bare command, optionally with arguments.
    if let Some((name, args)) = command_with_args(&t) {
        return Intent::Command {
            name: name.to_string(),
            args,
        };
    }

    Intent::Unknown
}

/// Lowercase, collapse whitespace, and strip punctuation that carries no meaning.
///
/// "Draw   a LINE, from  0,0" becomes "draw a line from 0 0", so the keyword
/// matcher never has to consider capitalisation or spacing.
fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        match c {
            // Punctuation that separates tokens becomes a space, so "0,0" and
            // "0 0" and "0. 0" all tokenize the same way.
            //
            // Parentheses are deliberately absent: they are structure in
            // arithmetic, and "2 * (3 + 4)" normalising to "2 * 3 + 4" is a
            // wrong answer rather than a differently-spelled right one.
            ',' | ';' | ':' | '[' | ']' | '{' | '}' => {
                if !prev_space {
                    out.push(' ');
                    prev_space = true;
                }
            }
            _ => {
                out.push(c.to_ascii_lowercase());
                prev_space = false;
            }
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Numbers that name a unit, so "5 mm" and "5" are both a length but only the
/// first carries one.
const UNITS: &[(&str, &str)] = &[
    ("mm", "mm"),
    ("millimetre", "mm"),
    ("cm", "cm"),
    ("m", "m"),
    ("in", "\""),
    ("inch", "\""),
    ("inches", "\""),
    ("ft", "'"),
    ("feet", "'"),
];

/// Find a shape in the sentence.
fn geometry(t: &str) -> Option<Intent> {
    let words: Vec<&str> = t.split_whitespace().collect();

    // "circle at x,y radius r" / "circle x y r"
    for (i, w) in words.iter().enumerate() {
        if *w != "circle" {
            continue;
        }
        let rest: Vec<&str> = words[i + 1..].to_vec();
        let mut pts = Vec::new();
        let mut radius = None;
        let mut skip = 0usize;
        for (j, w) in rest.iter().enumerate() {
            if skip > 0 {
                skip -= 1;
                continue;
            }
            if *w == "radius" || *w == "r" {
                if let Some(n) = number_after(&rest[j + 1..]) {
                    radius = Some(n);
                    skip = 1;
                }
                continue;
            }
            if *w == "at" || *w == "center" || *w == "centre" || *w == "from" {
                continue;
            }
            if let Some(p) = two_numbers(w) {
                pts.push(p);
                continue;
            }
            if let Some(n) = one_number(w) {
                // A bare number after "at" is the x of a centre point.
                if j > 0
                    && (rest[j - 1] == "at" || rest[j - 1] == "from")
                    && let Some(y) = rest.get(j + 1).and_then(|s| one_number(s))
                {
                    pts.push([n, y]);
                    skip = 1;
                }
            }
        }
        // A circle needs a centre and a radius, in either order.
        if let (Some(c), Some(r)) = (pts.first(), radius)
            && r > 0.0
        {
            return Some(Intent::Geometry {
                kind: GeometryKind::Circle,
                a: *c,
                b: [c[0] + r, c[1]],
            });
        }
    }

    // "line from a to b" / "line a b" / "line a b c d" (four numbers)
    for (i, w) in words.iter().enumerate() {
        if *w != "line" && *w != "segment" {
            continue;
        }
        let rest: Vec<&str> = words[i + 1..].to_vec();
        let mut skip = 0usize;
        let mut nums = Vec::new();
        for (j, w) in rest.iter().enumerate() {
            if skip > 0 {
                skip -= 1;
                continue;
            }
            match *w {
                "from" | "to" | "between" | "and" => continue,
                "length" | "len" => {
                    if let Some(n) = number_after(&rest[j + 1..]) {
                        nums.push(n);
                        skip = 1;
                    }
                }
                _ => {
                    if let Some([x, y]) = two_numbers(w) {
                        nums.extend_from_slice(&[x, y]);
                    } else if let Some(n) = one_number(w) {
                        nums.push(n);
                    }
                }
            }
        }
        // Two points, or a length with a start point.
        if nums.len() >= 4 {
            return Some(Intent::Geometry {
                kind: GeometryKind::Line,
                a: [nums[0], nums[1]],
                b: [nums[2], nums[3]],
            });
        }
        if nums.len() == 2 {
            return Some(Intent::Geometry {
                kind: GeometryKind::Line,
                a: [nums[0], nums[1]],
                b: [nums[0] + 10.0, nums[1]],
            });
        }
        if let Some(len) = nums.first() {
            return Some(Intent::Geometry {
                kind: GeometryKind::Line,
                a: [0.0, 0.0],
                b: [*len, 0.0],
            });
        }
    }

    // "rectangle from a to b" / "rect a b c d"
    for (i, w) in words.iter().enumerate() {
        if *w != "rectangle" && *w != "rect" && *w != "box" {
            continue;
        }
        let rest: Vec<&str> = words[i + 1..].to_vec();
        let mut nums = Vec::new();
        for w in rest.iter() {
            match *w {
                "from" | "to" | "between" | "and" | "at" | "with" | "size" => continue,
                _ => {
                    if let Some([x, y]) = two_numbers(w) {
                        nums.extend_from_slice(&[x, y]);
                    } else if let Some(n) = one_number(w) {
                        nums.push(n);
                    }
                }
            }
        }
        if nums.len() >= 4 {
            return Some(Intent::Geometry {
                kind: GeometryKind::Rectangle,
                a: [nums[0], nums[1]],
                b: [nums[2], nums[3]],
            });
        }
    }

    None
}

/// Does this sentence name a shape at all?
///
/// Separate from [`geometry`] so a shape keyword with its required arguments
/// missing is distinguishable from a sentence that never mentioned one. Without
/// that, "circle at 5,5" would fall through to the command branch and run
/// `circle`, putting a circle of radius 0 in the drawing.
fn names_a_shape(t: &str) -> bool {
    t.split_whitespace().any(|w| {
        matches!(
            w,
            "line" | "circle" | "arc" | "rectangle" | "rect" | "box" | "polyline"
        )
    })
}

/// Format a number the way a drawing reads it: no trailing zeros, and no
/// scientific notation for the magnitudes a plot scale uses.
fn format_number(v: f32) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    let s = format!("{v:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// The command a sentence names, plus whatever followed it.
fn command_with_args(t: &str) -> Option<(&str, String)> {
    // Longest match first. "zoom all" is one command: trying "zoom" alone would
    // either miss it or run the wrong thing, and "save as <name>" has to resolve
    // to `saveas` or the name is left as an argument to `save`.
    //
    // The phrase is a `String` because the words are borrowed from `t` while the
    // lookup returns a borrow of the *table*, so the returned command is
    // `'static` and the local phrase can be dropped at the end of the call.
    let words: Vec<&str> = t.split_whitespace().collect();
    for len in [3usize, 2, 1] {
        if words.len() < len {
            continue;
        }
        let phrase = words[..len].join(" ");
        if let Some(cmd) = lookup_command(&phrase) {
            let rest: Vec<&str> = words[len..].to_vec();
            return Some((cmd, rest.join(" ")));
        }
    }
    None
}

/// Resolve a word to a command, through aliases.
///
/// Returns `None` for anything unknown, which is what makes the caller able to
/// distinguish "this is a command" from "this is not a word I know" -- a
/// suggestion engine that accepts everything is decoration.
fn lookup_command(word: &str) -> Option<&'static str> {
    // Direct hit, then aliases. Both tables are `&'static`, so the result is
    // `'static` too -- which is what lets `command_with_args` build a local
    // phrase and still return a command that outlives it.
    if let Some(c) = VOCAB.iter().find(|c| **c == word) {
        return Some(c);
    }
    ALIASES
        .iter()
        .find(|(alias, _)| *alias == word)
        .map(|(_, canonical)| *canonical)
}

/// Command aliases, longest-first so "dim" does not shadow "dimlinear".
/// Multi-word entries are what make "zoom all" and "save as" work: the parser
/// tries the longest phrase first, so these have to be here or the single-word
/// lookup runs "zoom" and leaves "all" as an argument.
const ALIASES: &[(&str, &str)] = &[
    ("zoom all", "zoomall"),
    ("zoom in", "zoomin"),
    ("zoom out", "zoomout"),
    ("save as", "saveas"),
    ("dim linear", "dimlinear"),
    ("dim aligned", "dimaligned"),
    ("dim radius", "dimradius"),
    ("dim diameter", "dimdiameter"),
    ("dim angular", "dimangular"),
    ("view 3d", "view3d"),
    ("3d orbit", "view3d"),
    ("dimaligned", "dimaligned"),
    ("dimlinear", "dimlinear"),
    ("dimdiameter", "dimdiameter"),
    ("dimradius", "dimradius"),
    ("dimangular", "dimangular"),
    ("rect", "rectangle"),
    ("circle", "circle"),
    ("line", "line"),
    ("zoomall", "zoomall"),
    ("zoomin", "zoomin"),
    ("zoomout", "zoomout"),
    ("offset", "offset"),
    ("erase", "erase"),
    ("copy", "copytool"),
    ("move", "move"),
    ("trim", "trim"),
    ("extend", "extend"),
    ("mirror", "mirror"),
    ("rotate", "rotate"),
    ("block", "block"),
    ("insert", "insert"),
    ("explode", "explode"),
    ("save", "save"),
    ("saveas", "saveas"),
    ("open", "open"),
    ("export", "export"),
    ("plot", "export"),
    ("print", "export"),
    ("undo", "undo"),
    ("redo", "redo"),
    ("help", "help"),
    ("about", "about"),
];

/// One number out of a word, with an optional unit suffix.
fn one_number(s: &str) -> Option<f32> {
    // Split the longest leading run of digits and separators.
    let s = s.trim();
    let end = s
        .char_indices()
        .take_while(|(_, c)| c.is_ascii_digit() || *c == '.' || *c == '-')
        .last()
        .map(|(i, _)| i + 1)
        .unwrap_or(0);
    if end == 0 {
        return None;
    }
    s[..end].parse::<f32>().ok().or_else(|| {
        // "5mm" parses as a number once the letters go; "mm5" is not one.
        s[..end].parse::<f32>().ok()
    })
}

/// Two numbers out of a word, either "1,2" or "1x2" or "1 2".
fn two_numbers(s: &str) -> Option<[f32; 2]> {
    for sep in [',', 'x', 'X', '*', ':', ' '] {
        let parts: Vec<&str> = s.split(sep).collect();
        if parts.len() == 2
            && let (Some(a), Some(b)) = (one_number(parts[0]), one_number(parts[1]))
        {
            return Some([a, b]);
        }
    }
    None
}

/// The first parseable number in `words`, ignoring unit words.
fn number_after(words: &[&str]) -> Option<f32> {
    words
        .iter()
        .find(|w| !UNITS.iter().any(|(u, _)| *u == **w))
        .and_then(|w| one_number(w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_command_round_trips() {
        let i = parse("zoom all");
        assert_eq!(
            i,
            Intent::Command {
                name: "zoomall".to_string(),
                args: String::new()
            },
            "{i:?}"
        );
    }

    #[test]
    fn a_command_with_arguments_keeps_them() {
        let i = parse("save as plan.dxf");
        assert_eq!(
            i,
            Intent::Command {
                name: "saveas".to_string(),
                args: "plan.dxf".into()
            }
        );
    }

    #[test]
    fn articles_are_skipped_so_the_command_is_found() {
        let i = parse("please draw a circle");
        // "draw" and "a" are not commands, so parsing stops at "draw" and
        // returns Unknown -- which is correct: "draw" is not a command name.
        assert_eq!(i, Intent::Unknown, "{i:?}");
    }

    #[test]
    fn a_line_from_two_points() {
        let i = parse("line from 0,0 to 10,10");
        assert_eq!(
            i,
            Intent::Geometry {
                kind: GeometryKind::Line,
                a: [0.0, 0.0],
                b: [10.0, 10.0]
            },
            "{i:?}"
        );
    }

    #[test]
    fn a_line_from_four_numbers() {
        let i = parse("line 0 0 10 10");
        assert_eq!(
            i,
            Intent::Geometry {
                kind: GeometryKind::Line,
                a: [0.0, 0.0],
                b: [10.0, 10.0]
            },
            "{i:?}"
        );
    }

    #[test]
    fn a_circle_with_a_radius() {
        let i = parse("circle at 5,5 radius 3");
        match i {
            Intent::Geometry {
                kind: GeometryKind::Circle,
                a,
                ..
            } => {
                assert!((a[0] - 5.0).abs() < 1e-4, "{a:?}");
                assert!((a[1] - 5.0).abs() < 1e-4, "{a:?}");
            }
            other => panic!("expected a circle, got {other:?}"),
        }
    }

    #[test]
    fn a_circle_without_a_radius_is_rejected() {
        // Guessing a radius would put a circle in the drawing the user did not ask
        // for. Unknown is the honest answer.
        assert_eq!(parse("circle at 5,5"), Intent::Unknown);
    }

    #[test]
    fn a_rectangle_from_two_corners() {
        let i = parse("rectangle from 0,0 to 20,10");
        assert_eq!(
            i,
            Intent::Geometry {
                kind: GeometryKind::Rectangle,
                a: [0.0, 0.0],
                b: [20.0, 10.0]
            },
            "{i:?}"
        );
    }

    #[test]
    fn nonsense_is_unknown_not_a_guess() {
        for s in ["", "   ", "frobnicate the gizmo", "draw me a nice house"] {
            assert_eq!(parse(s), Intent::Unknown, "{s:?}");
        }
    }

    #[test]
    fn punctuation_does_not_change_the_tokens() {
        assert_eq!(
            normalize("Draw   a LINE, from  0,0"),
            "draw a line from 0 0"
        );
        assert_eq!(normalize(""), "");
        assert_eq!(normalize(" , ; : "), "");
        assert_eq!(normalize("CIRCLE"), "circle");
        assert_eq!(normalize("5 mm"), "5 mm");
    }

    #[test]
    fn units_are_recognised_as_suffixes() {
        let n = one_number("5mm").expect("5mm is a number");
        assert!((n - 5.0).abs() < 1e-5, "{n}");
        assert_eq!(one_number("mm"), None);
        assert_eq!(one_number("5"), Some(5.0));
        assert_eq!(one_number(""), None);
        assert_eq!(one_number("-3"), Some(-3.0));
    }

    #[test]
    fn a_pair_of_coordinates_separates_on_any_separator() {
        assert_eq!(two_numbers("1,2"), Some([1.0, 2.0]));
        assert_eq!(two_numbers("1x2"), Some([1.0, 2.0]));
        assert_eq!(two_numbers("1:2"), Some([1.0, 2.0]));
        assert_eq!(two_numbers("1 2"), Some([1.0, 2.0]));
        assert_eq!(two_numbers("1"), None);
        assert_eq!(two_numbers("abc"), None);
    }

    #[test]
    fn arithmetic_is_a_command_so_formulas_stop_failing() {
        // The command line doubles as a calculator, and the "=" prefix has to win
        // over the command parser or every formula would be a "command not found".
        assert_eq!(
            parse("= 5 + 3"),
            Intent::Command {
                name: "print".to_string(),
                args: "8".into()
            }
        );
        assert_eq!(
            parse("= 10/4"),
            Intent::Command {
                name: "print".to_string(),
                args: "2.5".into()
            }
        );
        assert_eq!(
            parse("= 2 * (3 + 4)"),
            Intent::Command {
                name: "print".to_string(),
                args: "14".into()
            }
        );
    }

    #[test]
    fn an_unknown_word_does_not_borrow_a_neighbours_command() {
        // The parser stops at the first non-command word. That is what keeps
        // "please erase the line" from running "erase" because "line" is a command.
        assert_eq!(parse("please erase the line"), Intent::Unknown);
    }

    #[test]
    fn aliases_resolve_to_real_commands() {
        assert_eq!(lookup_command("line"), Some("line"));
        assert_eq!(lookup_command("rect"), Some("rectangle"));
        assert_eq!(lookup_command("plot"), Some("export"));
        assert_eq!(lookup_command("dim"), None, "a bare dim is not a command");
        assert_eq!(lookup_command("zzzz"), None);
    }

    #[test]
    fn every_alias_points_at_a_command_that_exists() {
        // A dangling alias is a suggestion that runs nothing. `cad-ai` cannot
        // depend on `cad-app` (that would be a cycle: the app trains its model
        // from this crate), so the target has to be resolvable from here -- which
        // means either in the vocabulary or as another alias's target.
        let known: std::collections::HashSet<&str> = VOCAB
            .iter()
            .copied()
            .chain(ALIASES.iter().map(|(_, c)| *c))
            .collect();
        for (alias, canonical) in ALIASES {
            assert!(
                known.contains(canonical),
                "{alias} points at {canonical}, which is not a command"
            );
        }
    }

    #[test]
    fn every_geometry_kind_maps_to_a_command() {
        for k in [
            GeometryKind::Line,
            GeometryKind::Circle,
            GeometryKind::Rectangle,
        ] {
            assert!(VOCAB.contains(&k.command()), "{} has no command", k.noun());
        }
    }

    #[test]
    fn quantities_parse_with_and_without_units() {
        assert_eq!(one_number("5mm"), Some(5.0));
        assert_eq!(one_number("5"), Some(5.0));
        // A unit word on its own is not a number.
        assert_eq!(one_number("mm"), None);
    }

    #[test]
    fn a_negative_coordinate_is_a_number() {
        assert_eq!(one_number("-3"), Some(-3.0));
        assert_eq!(two_numbers("-3,-4"), Some([-3.0, -4.0]));
    }

    #[test]
    fn a_zero_radius_circle_is_not_geometry() {
        assert_eq!(parse("circle at 5,5 radius 0"), Intent::Unknown);
    }

    #[test]
    fn a_line_alone_defaults_to_a_unit_length() {
        // "line 5" is a 5-unit line from the origin. Guessing is right here
        // because a single number can only mean a length.
        let i = parse("line 5");
        assert_eq!(
            i,
            Intent::Geometry {
                kind: GeometryKind::Line,
                a: [0.0, 0.0],
                b: [5.0, 0.0]
            },
            "{i:?}"
        );
    }
}
