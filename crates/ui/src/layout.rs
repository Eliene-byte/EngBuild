//! Rectangles, sizing and the layout cursor.

use cad_core::Vec2;

pub use cad_core::Rect2 as Rect;

/// A width/height pair.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Size {
    pub w: f32,
    pub h: f32,
}

impl Size {
    pub const ZERO: Self = Self { w: 0.0, h: 0.0 };
    pub const fn new(w: f32, h: f32) -> Self {
        Self { w, h }
    }
    pub fn to_vec(self) -> Vec2 {
        Vec2::new(self.w, self.h)
    }
    pub fn from_vec(v: Vec2) -> Self {
        Self { w: v.x, h: v.y }
    }
    /// Grow to fit `self` and `other`.
    pub fn union(self, o: Size) -> Size {
        Size::new(self.w.max(o.w), self.h.max(o.h))
    }
    pub fn is_empty(self) -> bool {
        self.w <= 0.0 || self.h <= 0.0
    }
}

/// Horizontal / vertical alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Align {
    #[default]
    Start,
    Center,
    End,
}

/// Spacing on all four sides.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Padding {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

impl Padding {
    pub const ZERO: Self = Self {
        left: 0.0,
        right: 0.0,
        top: 0.0,
        bottom: 0.0,
    };
    pub const fn all(v: f32) -> Self {
        Self {
            left: v,
            right: v,
            top: v,
            bottom: v,
        }
    }
    pub fn symmetric(h: f32, v: f32) -> Self {
        Self {
            left: h,
            right: h,
            top: v,
            bottom: v,
        }
    }
    pub fn horizontal(self) -> f32 {
        self.left + self.right
    }
    pub fn vertical(self) -> f32 {
        self.top + self.bottom
    }
    pub fn grow(self, r: Rect) -> Rect {
        Rect::new(
            Vec2::new(r.min.x - self.left, r.min.y - self.top),
            Vec2::new(r.max.x + self.right, r.max.y + self.bottom),
        )
    }
    pub fn shrink(self, r: Rect) -> Rect {
        Rect::new(
            Vec2::new(r.min.x + self.left, r.min.y + self.top),
            Vec2::new(r.max.x - self.right, r.max.y - self.bottom),
        )
    }
}

/// A cursor that hands out rectangles top-down, the way every CAD panel is
/// arranged. Horizontal splitting is done by nesting `Layout`s.
#[derive(Debug, Clone)]
pub struct Layout {
    area: Rect,
    cursor_y: f32,
    row_height: f32,
    gap: f32,
    padding: Padding,
}

impl Layout {
    pub fn new(area: Rect) -> Self {
        let mut l = Self {
            area,
            cursor_y: 0.0,
            row_height: 24.0,
            gap: 4.0,
            padding: Padding::ZERO,
        };
        l.reset();
        l
    }

    pub fn with_padding(mut self, p: Padding) -> Self {
        self.padding = p;
        self.reset();
        self
    }
    pub fn with_row_height(mut self, h: f32) -> Self {
        self.row_height = h;
        self
    }
    pub fn with_gap(mut self, g: f32) -> Self {
        self.gap = g;
        self
    }

    /// Go back to the top of the area.
    pub fn reset(&mut self) {
        self.cursor_y = self.area.min.y + self.padding.top;
    }

    pub fn area(&self) -> Rect {
        self.area
    }
    /// Remaining vertical space below the cursor.
    pub fn remaining(&self) -> f32 {
        (self.area.max.y - self.padding.bottom - self.cursor_y).max(0.0)
    }

    /// Claim a full-width row of `h` height.
    pub fn row(&mut self, h: f32) -> Rect {
        let r = Rect::from_xywh(
            self.area.min.x + self.padding.left,
            self.cursor_y,
            self.area.width() - self.padding.horizontal(),
            h,
        );
        self.cursor_y += h + self.gap;
        r
    }

    /// Claim a row of the default height.
    pub fn next_row(&mut self) -> Rect {
        self.row(self.row_height)
    }

    /// Claim several equal columns within one row, honouring `align`.
    pub fn columns(&mut self, n: usize, gap: f32, align: Align) -> Vec<Rect> {
        if n == 0 {
            return Vec::new();
        }
        let y = self.cursor_y;
        self.cursor_y += self.row_height + self.gap;
        let total = self.area.width() - self.padding.horizontal();
        let w = ((total - gap * (n as f32 - 1.0)) / n as f32).max(0.0);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let x = match align {
                Align::Start => self.area.min.x + self.padding.left + i as f32 * (w + gap),
                Align::Center => {
                    let used = n as f32 * w + (n as f32 - 1.0) * gap;
                    let x0 = self.area.min.x + self.padding.left + (total - used) * 0.5;
                    x0 + i as f32 * (w + gap)
                }
                Align::End => {
                    let used = n as f32 * w + (n as f32 - 1.0) * gap;
                    let x0 = self.area.min.x + self.padding.left + (total - used);
                    x0 + i as f32 * (w + gap)
                }
            };
            out.push(Rect::from_xywh(x, y, w, self.row_height));
        }
        out
    }

    /// Split the area into `n` vertical panes. The first pane gets
    /// `sizes[0] / total` of the width, the last absorbs the remainder so
    /// rounding never leaves a gap.
    pub fn split_vertical(&self, sizes: &[f32]) -> Vec<Rect> {
        if sizes.is_empty() {
            return Vec::new();
        }
        let total: f32 = sizes.iter().sum();
        if total <= 0.0 {
            return vec![self.area; sizes.len()];
        }
        let mut out = Vec::with_capacity(sizes.len());
        let mut x = self.area.min.x;
        for (i, s) in sizes.iter().enumerate() {
            let w = if i + 1 == sizes.len() {
                self.area.max.x - x
            } else {
                (self.area.width() * s / total).max(0.0)
            };
            out.push(Rect::from_xywh(x, self.area.min.y, w, self.area.height()));
            x += w;
        }
        out
    }

    /// Split the area into `n` horizontal strips of equal height.
    pub fn split_horizontal(&self, n: usize) -> Vec<Rect> {
        if n == 0 {
            return Vec::new();
        }
        let h = self.area.height() / n as f32;
        (0..n)
            .map(|i| {
                let y = self.area.min.y + i as f32 * h;
                let hh = if i + 1 == n { self.area.max.y - y } else { h };
                Rect::from_xywh(self.area.min.x, y, self.area.width(), hh)
            })
            .collect()
    }

    /// Skip `h` pixels vertically.
    pub fn space(&mut self, h: f32) {
        self.cursor_y += h;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area() -> Rect {
        Rect::from_xywh(0.0, 0.0, 200.0, 300.0)
    }

    #[test]
    fn rows_stack_downwards() {
        let mut l = Layout::new(area()).with_row_height(20.0).with_gap(0.0);
        let a = l.next_row();
        let b = l.next_row();
        assert_eq!(a.min.y, 0.0);
        assert_eq!(b.min.y, 20.0);
        assert_eq!(a.width(), 200.0);
    }

    #[test]
    fn gap_is_applied_between_rows() {
        let mut l = Layout::new(area()).with_row_height(20.0).with_gap(5.0);
        l.next_row();
        assert_eq!(l.next_row().min.y, 25.0);
    }

    #[test]
    fn padding_insets_rows_and_remaining() {
        let mut l = Layout::new(area()).with_padding(Padding::all(10.0));
        let r = l.next_row();
        assert_eq!(r.min.x, 10.0);
        assert_eq!(r.min.y, 10.0);
        assert_eq!(r.width(), 180.0);
        // 300 - 20 padding - one 24px row - 4px gap
        assert!(
            (l.remaining() - (300.0 - 20.0 - 24.0 - 4.0)).abs() < 1e-4,
            "{}",
            l.remaining()
        );
    }

    #[test]
    fn reset_returns_to_the_top() {
        let mut l = Layout::new(area());
        l.next_row();
        l.next_row();
        l.reset();
        assert_eq!(l.next_row().min.y, 0.0);
    }

    #[test]
    fn columns_tile_the_row() {
        let mut l = Layout::new(area()).with_gap(10.0);
        let cols = l.columns(3, 10.0, Align::Start);
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].min.x, 0.0);
        assert_eq!(cols[1].min.x, cols[0].width() + 10.0);
        assert_eq!(
            cols[2].max.x, 200.0,
            "last column must reach the right edge"
        );
        for c in &cols {
            assert!((c.width() - (200.0 - 20.0) / 3.0).abs() < 1e-4, "{c:?}");
        }
    }

    #[test]
    fn columns_with_centre_alignment_are_centred() {
        let mut l = Layout::new(area());
        let cols = l.columns(2, 0.0, Align::Center);
        assert!(cols[0].min.x > 0.0);
        assert!(cols[1].max.x < 200.0);
        assert!((cols[0].min.x - 50.0).abs() < 1e-4);
    }

    #[test]
    fn split_vertical_covers_exactly() {
        let l = Layout::new(area());
        let panes = l.split_vertical(&[1.0, 1.0, 2.0]);
        assert_eq!(panes.len(), 3);
        assert_eq!(panes[0].min.x, 0.0);
        assert_eq!(panes[2].max.x, 200.0);
        assert!((panes[0].max.x - panes[1].min.x).abs() < 1e-4);
        assert!((panes[1].max.x - panes[2].min.x).abs() < 1e-4);
    }

    #[test]
    fn split_vertical_handles_degenerate_input() {
        let l = Layout::new(area());
        assert!(l.split_vertical(&[]).is_empty());
        let panes = l.split_vertical(&[0.0]);
        assert_eq!(panes.len(), 1);
        assert_eq!(
            panes[0].width(),
            200.0,
            "zero fraction still fills the area"
        );
    }

    #[test]
    fn split_horizontal_tiles_evenly() {
        let l = Layout::new(area());
        let rows = l.split_horizontal(3);
        assert_eq!(rows.len(), 3);
        assert!((rows[0].height() - 100.0).abs() < 1e-4);
        assert_eq!(rows[2].max.y, 300.0);
    }

    #[test]
    fn padding_grow_and_shrink_are_inverses() {
        let r = area();
        let p = Padding::symmetric(8.0, 4.0);
        let grown = p.grow(r);
        assert_eq!(grown.min.x, -8.0);
        assert_eq!(grown.max.y, 304.0);
        let shrunk = p.shrink(grown);
        assert!(shrunk.min.distance(r.min) < 1e-5);
        assert!(shrunk.max.distance(r.max) < 1e-5);
    }

    #[test]
    fn size_union_and_vec_conversions() {
        let a = Size::new(10.0, 20.0);
        assert_eq!(a.union(Size::new(15.0, 5.0)), Size::new(15.0, 20.0));
        assert_eq!(Size::from_vec(a.to_vec()), a);
        assert!(Size::ZERO.is_empty());
        assert!(!a.is_empty());
    }

    #[test]
    fn space_skips_pixels() {
        let mut l = Layout::new(area());
        l.space(30.0);
        assert_eq!(l.next_row().min.y, 30.0);
    }
}
