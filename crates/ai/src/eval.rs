//! A tiny arithmetic evaluator for the command line.
//!
//! The command line doubles as a calculator, which is how every CAD app built
//! before scripting worked: `= 5 + 3` gives 8, and `= 2 * (3 + 4)` gives 14.
//!
//! Recursive descent, no dependencies, no allocation beyond the input slice.
//! Precedence is the one everyone expects: parentheses, then `*` and `/`, then
//! `+` and `-`, with unary minus binding tighter than `*` so `-2 * 3` is -6.

/// Evaluate `expr`, or `None` when it is not a well-formed expression.
///
/// Whitespace is ignored. An empty string is not an error, just nothing to do.
pub fn evaluate(expr: &str) -> Option<f32> {
    let mut p = Parser {
        bytes: expr.as_bytes(),
        pos: 0,
    };
    let v = p.expr()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        // Trailing garbage: "5 + 3 x" is not an expression, and silently
        // returning 5+3 would be worse than saying so.
        return None;
    }
    if !v.is_finite() {
        return None;
    }
    Some(v)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ') | Some(b'\t') | Some(b'\n')) {
            self.pos += 1;
        }
    }

    /// `term (("+" | "-") term)*`
    fn expr(&mut self) -> Option<f32> {
        let mut left = self.term()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    left += self.term()?;
                }
                Some(b'-') => {
                    self.pos += 1;
                    left -= self.term()?;
                }
                _ => return Some(left),
            }
        }
    }

    /// `factor (("*" | "/") factor)*`
    fn term(&mut self) -> Option<f32> {
        let mut left = self.factor()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    left *= self.factor()?;
                }
                Some(b'/') => {
                    self.pos += 1;
                    let d = self.factor()?;
                    if d == 0.0 {
                        // Division by zero is not a number, and returning
                        // infinity would put a nonsense value in the drawing.
                        return None;
                    }
                    left /= d;
                }
                _ => return Some(left),
            }
        }
    }

    /// A number, a parenthesised expression, or a unary sign.
    fn factor(&mut self) -> Option<f32> {
        self.skip_ws();
        match self.peek()? {
            b'(' => {
                self.pos += 1;
                let v = self.expr()?;
                self.skip_ws();
                if self.peek() != Some(b')') {
                    return None;
                }
                self.pos += 1;
                Some(v)
            }
            b'-' => {
                self.pos += 1;
                Some(-self.factor()?)
            }
            b'+' => {
                self.pos += 1;
                self.factor()
            }
            _ => self.number(),
        }
    }

    /// A decimal number, integer or not.
    fn number(&mut self) -> Option<f32> {
        self.skip_ws();
        let start = self.pos;
        let mut seen_dot = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.pos += 1;
            } else if c == b'.' && !seen_dot {
                seen_dot = true;
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return None;
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).ok()?;
        s.parse::<f32>().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_obvious_arithmetic() {
        assert_eq!(evaluate("5 + 3"), Some(8.0));
        assert_eq!(evaluate("10 - 4"), Some(6.0));
        assert_eq!(evaluate("6 * 7"), Some(42.0));
        assert_eq!(evaluate("10 / 4"), Some(2.5));
    }

    #[test]
    fn precedence_is_the_one_everyone_expects() {
        assert_eq!(evaluate("2 + 3 * 4"), Some(14.0));
        assert_eq!(evaluate("2 * 3 + 4"), Some(10.0));
        assert_eq!(evaluate("10 - 2 - 3"), Some(5.0), "left associative");
        assert_eq!(evaluate("100 / 10 / 2"), Some(5.0));
    }

    #[test]
    fn parentheses_override_precedence() {
        assert_eq!(evaluate("(2 + 3) * 4"), Some(20.0));
        assert_eq!(evaluate("2 * (3 + 4)"), Some(14.0));
        assert_eq!(evaluate("((1 + 2))"), Some(3.0));
    }

    #[test]
    fn unary_minus_binds_tighter_than_multiplication() {
        // This is the case that gets it wrong: -2 * 3 is -6, not -(2*3) which
        // happens to agree, so check the one that does not.
        assert_eq!(evaluate("-2 + 3"), Some(1.0));
        assert_eq!(evaluate("3 * -2"), Some(-6.0));
        assert_eq!(evaluate("-(2 + 3)"), Some(-5.0));
        assert_eq!(evaluate("--5"), Some(5.0));
    }

    #[test]
    fn whitespace_is_ignored() {
        assert_eq!(evaluate("  5  +  3  "), Some(8.0));
        assert_eq!(evaluate(""), None);
        assert_eq!(evaluate("   "), None);
    }

    #[test]
    fn decimals_work() {
        assert!((evaluate("0.5 + 0.25").unwrap() - 0.75).abs() < 1e-6);
        assert!((evaluate("1.5 * 2").unwrap() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn trailing_garbage_is_rejected_not_truncated() {
        // Silently returning 8 for "5 + 3 x" would put a wrong number in a
        // drawing, which is worse than saying the expression is malformed.
        assert_eq!(evaluate("5 + 3 x"), None);
        assert_eq!(evaluate("5 3"), None);
        assert_eq!(evaluate("(5 + 3"), None);
        assert_eq!(evaluate("5 + )"), None);
    }

    #[test]
    fn division_by_zero_is_not_a_number() {
        assert_eq!(evaluate("5 / 0"), None);
        assert_eq!(evaluate("1 / (2 - 2)"), None);
    }

    #[test]
    fn a_single_number_is_an_expression() {
        assert_eq!(evaluate("42"), Some(42.0));
        assert_eq!(evaluate("-42"), Some(-42.0));
    }

    #[test]
    fn a_long_expression_stays_accurate() {
        // 1 + 6 - 2 + 8 = 13. The point of the test is that the whole expression
        // is evaluated, not that the answer is a round number.
        let v = evaluate("1 + 2 * 3 - 4 / 2 + (5 - 1) * 2").expect("parses");
        assert!((v - 13.0).abs() < 1e-4, "{v}");
    }

    #[test]
    fn an_empty_parenthesis_is_not_zero() {
        assert_eq!(evaluate("()"), None);
        assert_eq!(evaluate("5 * ()"), None);
    }
}
