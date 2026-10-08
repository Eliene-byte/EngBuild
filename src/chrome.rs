//! The application chrome: menu bar, ribbon, layer panel, properties panel,
//! command line and status bar.
//!
//! This is the layer that was missing. `cad-ui` shipped a full widget toolkit and
//! the binary used none of it — the panels were background rectangles. Everything
//! here is drawn through [`cad_ui::Ui`], so a click goes through the same
//! hit-testing as the canvas does, and the panels show real document state.
//!
//! Widgets run in an immediate-mode pass: they only *propose* actions, which the
//! caller applies afterwards. That is what keeps the borrow checker happy while
//! still letting a layer button mutate the very table it is iterating.

use cad_app::session::{Session, StatusMessage, ViewMode};
use cad_core::{Rgba, Vec2};
use cad_doc::{EntityKind, LayerId};
use cad_snap::SnapKind;
use cad_ui::Ui;
use cad_ui::layout::Rect;
use cad_ui::text_width;
use cad_ui::widgets::TextEditState;

/// Something the user asked for, to be applied after the widget pass.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Run a named command through the command registry.
    Command(String),
    /// The command line was submitted.
    Submit(String),
    /// Accept the top completion (Tab, or End on the line).
    AcceptCompletion(String),
    ToggleLayerVisible(LayerId),
    ToggleLayerLocked(LayerId),
    SetCurrentLayer(LayerId),
    NewLayer,
    /// Discard unsaved changes and start over.
    NewDocument,
    Open,
    Save,
    SaveAs,
    Export,
    About,
}

/// Panel geometry in CSS pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Panels {
    pub canvas: Rect,
    pub menu_bar: Rect,
    pub ribbon: Rect,
    pub command_line: Rect,
    pub layer_panel: Rect,
    pub properties: Rect,
    pub status_bar: Rect,
}

impl Panels {
    /// Split the window the way a CAD app is laid out: a menu bar and ribbon on
    /// top, the command line and status bar at the bottom, panels on the sides.
    ///
    /// Every rect tiles the window exactly, with no overlap and no gaps, so the
    /// canvas is always precisely the region the camera is sized to.
    pub fn layout(w: f32, h: f32) -> Self {
        let w = w.max(0.0);
        let h = h.max(0.0);
        let menu_h = (24.0f32).min(h * 0.08);
        let ribbon_h = (104.0f32).min(h * 0.22);
        let cmd_h = (30.0f32).min(h * 0.1);
        let status_h = (24.0f32).min(h * 0.1);
        let side_w = (250.0f32).min(w * 0.24);

        let top = menu_h + ribbon_h;
        let body_bottom = (h - cmd_h - status_h).max(top);
        let body_h = body_bottom - top;

        Self {
            menu_bar: Rect::from_xywh(0.0, 0.0, w, menu_h),
            ribbon: Rect::from_xywh(0.0, menu_h, w, ribbon_h),
            canvas: Rect::from_xywh(side_w, top, (w - side_w * 2.0).max(0.0), body_h),
            layer_panel: Rect::from_xywh(0.0, top, side_w, body_h),
            properties: Rect::from_xywh((w - side_w).max(0.0), top, side_w, body_h),
            command_line: Rect::from_xywh(0.0, body_bottom, w, cmd_h),
            status_bar: Rect::from_xywh(0.0, (body_bottom + cmd_h).min(h), w, status_h),
        }
    }
}

/// Top-level menus, in the order the bar shows them.
pub const MENUS: [&str; 7] = ["File", "Edit", "View", "Draw", "Modify", "3D", "Help"];

/// Ribbon groups. Each is a labelled block of buttons.
struct RibbonGroup {
    title: &'static str,
    /// `(icon name, command, label, shortcut)`.
    ///
    /// The icon name comes first so a button cannot be added with a command
    /// and no icon, or with an icon no command runs.
    buttons: &'static [(&'static str, &'static str, &'static str, &'static str)],
}

const DRAW_GROUP: &[(&str, &str, &str, &str)] = &[
    ("line", "line", "Line", "L"),
    ("circle", "circle", "Circle", "C"),
    ("arc", "arc", "Arc", "A"),
    ("polyline", "polyline", "Polyline", "PL"),
    ("rectangle", "rectangle", "Rect", "REC"),
];

const MODIFY_GROUP: &[(&str, &str, &str, &str)] = &[
    ("move", "move", "Move", "M"),
    ("copy", "copytool", "Copy", "CP"),
    ("rotate", "rotate", "Rotate", "RO"),
    ("mirror", "mirror", "Mirror", "MI"),
    ("offset", "offset", "Offset", "O"),
    ("trim", "trim", "Trim", "TR"),
    ("trash", "erase", "Erase", "E"),
];

const VIEW_GROUP: &[(&str, &str, &str, &str)] = &[
    ("zoomall", "zoomall", "Zoom All", "Z"),
    ("zoomin", "zoomin", "Zoom In", ""),
    ("zoomout", "zoomout", "Zoom Out", ""),
    ("3d", "view3d", "3D Orbit", ""),
];

const FILE_GROUP: &[(&str, &str, &str, &str)] = &[
    ("new", "new", "New", "Ctrl+N"),
    ("open", "open", "Open", "Ctrl+O"),
    ("save", "save", "Save", "Ctrl+S"),
    ("export", "export", "Export", ""),
];

/// Which ribbon tab is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RibbonTab {
    Draw,
    Modify,
    View,
    File,
}

impl RibbonTab {
    pub const ALL: [RibbonTab; 4] = [
        RibbonTab::Draw,
        RibbonTab::Modify,
        RibbonTab::View,
        RibbonTab::File,
    ];
    pub fn label(self) -> &'static str {
        match self {
            RibbonTab::Draw => "Draw",
            RibbonTab::Modify => "Modify",
            RibbonTab::View => "View",
            RibbonTab::File => "File",
        }
    }
    /// Groups shown for this tab.
    fn groups(self) -> Vec<RibbonGroup> {
        match self {
            RibbonTab::Draw => vec![RibbonGroup {
                title: "Draw",
                buttons: DRAW_GROUP,
            }],
            RibbonTab::Modify => vec![
                RibbonGroup {
                    title: "Modify",
                    buttons: MODIFY_GROUP,
                },
                RibbonGroup {
                    title: "3D",
                    buttons: &[("extrude", "extrude", "Extrude", "")],
                },
            ],
            RibbonTab::View => vec![RibbonGroup {
                title: "View",
                buttons: VIEW_GROUP,
            }],
            RibbonTab::File => vec![RibbonGroup {
                title: "File",
                buttons: FILE_GROUP,
            }],
        }
    }
}

/// Persistent UI state that widgets cannot own: text buffers and panel scroll.
#[derive(Debug, Clone, Default)]
pub struct Chrome {
    /// The command line's editing state.
    pub command: TextEditState,
    /// Which ribbon tab is open.
    pub tab: Option<RibbonTab>,
    /// Index into [`MENUS`] of the open drop-down, if any.
    pub open_menu: Option<usize>,
    /// Layer the user last clicked in the layer panel.
    pub layer_scroll: f32,
    /// Whether the "Entity" section of the properties panel is expanded.
    pub show_entity: bool,
}

impl Chrome {
    pub fn new() -> Self {
        Self {
            command: TextEditState::new(""),
            tab: Some(RibbonTab::Draw),
            open_menu: None,
            layer_scroll: 0.0,
            show_entity: true,
        }
    }
}

/// What the suggestion model said, computed by the caller.
///
/// The widget pass only has `&Session`, and predicting needs `&mut Session`
/// because the model trains lazily, so the answer is computed before the pass
/// and handed in rather than asked for inside it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Suggestions {
    /// Completions for what has been typed, best first.
    pub completions: Vec<String>,
    /// The model's top guess with an empty line.
    pub next: Option<&'static str>,
    /// The tail of the top completion, to draw as ghost text.
    pub ghost: Option<String>,
}

impl Suggestions {
    /// Build from a prefix and a model's output.
    pub fn build(prefix: &str, completions: Vec<String>, next: Option<&'static str>) -> Self {
        let ghost = completions
            .first()
            .filter(|c| {
                c.len() > prefix.len()
                    && c.to_ascii_lowercase()
                        .starts_with(&prefix.to_ascii_lowercase())
            })
            .map(|c| c[prefix.len()..].to_string());
        Self {
            completions,
            next,
            ghost,
        }
    }

    /// Is there a completion to accept?
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn has(&self) -> bool {
        !self.completions.is_empty()
    }
}

/// Read-only measurements the status bar shows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StatusFacts {
    pub cursor: Vec2,
    pub scale: f32,
    pub entities: usize,
    pub selection: usize,
    pub fps: f32,
    pub adapter: &'static str,
}

/// Draw the whole chrome and report what the user asked for.
///
/// `ui` borrows the input state and the two batches, so nothing here can mutate
/// the session; every interaction comes back as an [`Action`] for the caller to
/// apply. That is the whole point of immediate mode: the widget pass reads
/// state, the caller writes it.
pub fn draw(
    ui: &mut Ui<'_>,
    chrome: &mut Chrome,
    session: &Session,
    panels: Panels,
    facts: StatusFacts,
    suggestions: &Suggestions,
) -> Vec<Action> {
    let mut actions = Vec::new();
    draw_menu_bar(ui, chrome, session, panels, &mut actions);
    draw_ribbon(ui, chrome, session, panels, &mut actions);
    draw_layer_panel(ui, chrome, session, panels, &mut actions);
    draw_properties(ui, session, panels);
    draw_command_line(ui, chrome, session, panels, suggestions, &mut actions);
    draw_status_bar(ui, session, panels, facts, suggestions, &mut actions);
    actions
}

/// Panel background, drawn before the widgets in it.
fn panel_bg(ui: &mut Ui<'_>, r: Rect, color: Rgba) {
    ui.fill_rect(r, color);
}

/// A row of the menu bar, opening a drop-down when clicked.
fn draw_menu_bar(
    ui: &mut Ui<'_>,
    chrome: &mut Chrome,
    session: &Session,
    panels: Panels,
    out: &mut Vec<Action>,
) {
    let bar = panels.menu_bar;
    if bar.is_empty() {
        return;
    }
    panel_bg(ui, bar, ui.theme.surface);
    ui.fill_rect(
        Rect::from_xywh(bar.min.x, bar.max.y - 1.0, bar.width(), 1.0),
        ui.theme.border,
    );

    let pad = 10.0;
    let mut x = bar.min.x + 2.0;
    let text_col = ui.theme.text;
    let dim = ui.theme.text_dim;

    for (i, name) in MENUS.iter().enumerate() {
        let w = text_width(name, ui.theme.font_size) + pad * 2.0;
        let r = Rect::from_xywh(x, bar.min.y + 1.0, w, bar.height() - 2.0);
        let resp = ui.button(r, true);
        let open = chrome.open_menu == Some(i);
        if open {
            // Repaint the active item so the open menu is unmistakable.
            ui.fill_round_rect(r, ui.theme.accent, ui.theme.border_radius);
            ui.text_sized(
                name,
                Vec2::new(r.min.x + pad, r.center().y - ui.theme.font_size * 0.5),
                ui.theme.accent_text,
                ui.theme.font_size,
            );
        } else {
            ui.text_sized(
                name,
                Vec2::new(r.min.x + pad, r.center().y - ui.theme.font_size * 0.5),
                if resp.hovered { text_col } else { dim },
                ui.theme.font_size,
            );
        }
        if resp.clicked {
            chrome.open_menu = if open { None } else { Some(i) };
        }
        x += w;
    }

    // Drop-down, drawn last so it sits above everything to its right.
    if let Some(i) = chrome.open_menu
        && let Some(top) = MENUS.get(i)
    {
        let entries = session.commands.menu_entries(top);
        let row = ui.theme.row_height;
        let w: f32 = entries
            .iter()
            .filter_map(|e| session.commands.at(*e))
            .map(|c| text_width(c.label, ui.theme.font_size) + 44.0)
            .fold(120.0f32, f32::max);
        let h = row * entries.len().max(1) as f32 + 4.0;
        // Place it under the clicked label.
        let mut ax = bar.min.x + 2.0;
        for name in MENUS.iter().take(i) {
            ax += text_width(name, ui.theme.font_size) + 20.0;
        }
        let dd = Rect::from_xywh(ax, bar.max.y, w, h);
        ui.fill_round_rect(dd, ui.theme.surface, ui.theme.border_radius);
        ui.stroke_rect(dd, ui.theme.border, 1.0);

        for (k, idx) in entries.iter().enumerate() {
            let Some(cmd) = session.commands.at(*idx) else {
                continue;
            };
            let r = Rect::from_xywh(
                dd.min.x + 2.0,
                dd.min.y + 2.0 + k as f32 * row,
                dd.width() - 4.0,
                row,
            );
            let resp = ui.button(r, true);
            ui.text_sized(
                cmd.label,
                Vec2::new(r.min.x + 8.0, r.center().y - ui.theme.font_size * 0.5),
                ui.theme.text,
                ui.theme.font_size,
            );
            if let Some(sc) = cmd.shortcut {
                let sw = text_width(sc, ui.theme.font_size);
                ui.text_sized(
                    sc,
                    Vec2::new(r.max.x - sw - 8.0, r.center().y - ui.theme.font_size * 0.5),
                    ui.theme.text_dim,
                    ui.theme.font_size,
                );
            }
            if resp.clicked {
                out.push(Action::Command(cmd.name.to_string()));
                chrome.open_menu = None;
            }
        }
    }

    // Title on the right of the bar: file name plus a dirty marker.
    let title = title_of(session);
    let tw = text_width(&title, ui.theme.font_size);
    if tw + 20.0 < bar.width() {
        ui.text_sized(
            &title,
            Vec2::new(
                bar.max.x - tw - 10.0,
                bar.center().y - ui.theme.font_size * 0.5,
            ),
            ui.theme.text_dim,
            ui.theme.font_size,
        );
    }
}

fn title_of(session: &Session) -> String {
    let name = session
        .path
        .as_ref()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Untitled".to_string());
    if session.dirty {
        format!("{name} *")
    } else {
        name
    }
}

/// Tabs plus the active tab's button groups.
fn draw_ribbon(
    ui: &mut Ui<'_>,
    chrome: &mut Chrome,
    session: &Session,
    panels: Panels,
    out: &mut Vec<Action>,
) {
    let rib = panels.ribbon;
    if rib.is_empty() {
        return;
    }
    panel_bg(ui, rib, ui.theme.background);

    let tabs = ribbon_tabs_rects(rib, ui.theme);
    for (tab, r) in RibbonTab::ALL.iter().zip(tabs.iter()) {
        let active = chrome.tab == Some(*tab);
        let resp = ui.button(*r, true);
        if active {
            ui.fill_round_rect(*r, ui.theme.surface, ui.theme.border_radius);
        }
        let tw = text_width(tab.label(), ui.theme.font_size);
        ui.text_sized(
            tab.label(),
            Vec2::new(
                r.center().x - tw * 0.5,
                r.center().y - ui.theme.font_size * 0.5,
            ),
            if active {
                ui.theme.accent
            } else if resp.hovered {
                ui.theme.text
            } else {
                ui.theme.text_dim
            },
            ui.theme.font_size,
        );
        if resp.clicked {
            chrome.tab = Some(*tab);
        }
    }

    let Some(tab) = chrome.tab else {
        return;
    };
    let body = Rect::new(
        Vec2::new(rib.min.x + 4.0, tabs[0].max.y + 2.0),
        Vec2::new(rib.max.x - 4.0, rib.max.y - 2.0),
    );
    if body.height() < ui.theme.row_height {
        return;
    }
    let groups = tab.groups();
    // Space the groups evenly rather than letting the last one sprawl.
    let n = groups.len().max(1);
    let gap = 10.0;
    let gw = (body.width() - gap * (n as f32 - 1.0)) / n as f32;
    for (gi, g) in groups.iter().enumerate() {
        let gx = body.min.x + gi as f32 * (gw + gap);
        let gr = Rect::from_xywh(gx, body.min.y, gw, body.height());
        draw_ribbon_group(ui, g, gr, session, out);
    }
}

/// Tab strip geometry, factored out so the drawing and the layout cannot drift.
fn ribbon_tabs_rects(rib: Rect, theme: cad_ui::Theme) -> Vec<Rect> {
    // Each tab is as wide as its own label: a fixed width either clips "Modify"
    // or wastes space on "Draw". They are laid out left to right and must not
    // overlap, or only the leftmost one is ever clickable.
    let mut x = rib.min.x + 4.0;
    let y = rib.min.y + 2.0;
    RibbonTab::ALL
        .iter()
        .map(|t| {
            let w = text_width(t.label(), theme.font_size) + 18.0;
            let r = Rect::from_xywh(x, y, w, theme.row_height);
            x += w;
            r
        })
        .collect()
}

fn draw_ribbon_group(
    ui: &mut Ui<'_>,
    g: &RibbonGroup,
    area: Rect,
    session: &Session,
    out: &mut Vec<Action>,
) {
    // Group plate.
    ui.fill_round_rect(area, ui.theme.surface, ui.theme.border_radius);
    ui.stroke_rect(area, ui.theme.border, 1.0);

    let label = g.title;
    let lw = text_width(label, ui.theme.font_size) + 10.0;
    ui.text_sized(
        label,
        Vec2::new(
            area.center().x - lw * 0.5,
            area.max.y - ui.theme.font_size - 2.0,
        ),
        ui.theme.text_dim,
        ui.theme.font_size,
    );

    let btn_h = (area.height() - ui.theme.font_size - 6.0).max(ui.theme.row_height);
    let top = area.min.y + 3.0;
    let rows: Vec<Vec<&(&str, &str, &str, &str)>> =
        g.buttons.chunks(3).map(|c| c.iter().collect()).collect();
    let avail = area.width() - 6.0;
    let per_row = rows.len().max(1) as f32;
    let col_w = (avail / per_row.max(1.0) - 3.0).max(24.0);

    for (ri, row) in rows.iter().enumerate() {
        let y = top + ri as f32 * (btn_h + 2.0);
        if y + btn_h > area.max.y - ui.theme.font_size - 4.0 {
            break;
        }
        for (ci, (icon_name, cmd, label, shortcut)) in row.iter().enumerate() {
            let x = area.min.x + 3.0 + ci as f32 * (col_w + 3.0);
            let r = Rect::from_xywh(x, y, col_w, btn_h);
            let needs_sel = command_needs_selection(cmd, session);
            let enabled = !needs_sel;
            // The tool already showing is underlined, so the ribbon states which
            // command is live without the user reading the command line.
            let active = session.tool.id.command() == *cmd;
            let resp = ui.button(r, enabled);
            if let Some(ic) = cad_ui::icons::by_name(icon_name) {
                ui.icon(
                    ic,
                    r,
                    if active {
                        ui.theme.accent
                    } else if !enabled {
                        ui.theme.text_disabled
                    } else if resp.hovered {
                        ui.theme.text
                    } else {
                        ui.theme.text_dim
                    },
                    0.62,
                );
            }
            // An underline rather than a fill: the plate colour already carries
            // hover, and two states on the same plate makes both ambiguous.
            if active {
                ui.fill_rect(
                    Rect::from_xywh(r.min.x, r.max.y - 2.0, r.width(), 2.0),
                    ui.theme.accent,
                );
            }
            // A tip on every button. The icon is a reminder, the tooltip is the
            // documentation, and a button with neither is a guessing game.
            // Bind it to a name: `format!` in a temporary position inside a
            // `tooltip` argument is freed before the call returns.
            let tip = if !enabled {
                "Select objects first".to_string()
            } else if shortcut.is_empty() {
                label.to_string()
            } else {
                format!("{label} ({shortcut})")
            };
            ui.tooltip(r, &tip);
            if resp.clicked {
                out.push(Action::Command((*cmd).to_string()));
            }
        }
    }
}

/// True when `command` cannot run because nothing is selected.
fn command_needs_selection(command: &str, session: &Session) -> bool {
    cad_app::session::tool_for_command(command).is_some_and(|t| t.needs_selection())
        && session.tool.selection.is_empty()
}

/// The layer panel: every layer, its visibility and lock state, and the current one.
fn draw_layer_panel(
    ui: &mut Ui<'_>,
    chrome: &mut Chrome,
    session: &Session,
    panels: Panels,
    out: &mut Vec<Action>,
) {
    let p = panels.layer_panel;
    if p.is_empty() {
        return;
    }
    panel_bg(ui, p, ui.theme.background);

    let row = ui.theme.row_height;
    let head = Rect::from_xywh(p.min.x, p.min.y, p.width(), row + 6.0);
    // A header plate rather than bare text: a panel that reads as a surface of
    // its own says what it contains before the user looks for the label.
    ui.fill_rect(
        Rect::from_xywh(p.min.x, p.min.y, p.width(), head.height()),
        ui.theme.surface,
    );
    ui.rule(
        Rect::from_xywh(head.min.x + 8.0, head.min.y + 4.0, head.width() - 96.0, row),
        "LAYERS",
        ui.theme.text_dim,
    );
    // Icon-plus, not the words "+ New": at 26px wide the word wraps.
    if let Some(ic) = cad_ui::icon("plus") {
        let r = Rect::from_xywh(head.max.x - 34.0, head.min.y + 3.0, 30.0, row);
        let resp = ui.button(r, true);
        ui.icon(
            ic,
            r,
            if resp.hovered {
                ui.theme.text
            } else {
                ui.theme.accent
            },
            0.6,
        );
        ui.tooltip(r, "New layer");
        if resp.clicked {
            out.push(Action::NewLayer);
        }
    }
    ui.fill_rect(
        Rect::from_xywh(p.min.x, head.max.y, p.width(), 1.0),
        ui.theme.border,
    );

    let current = session.doc.current_layer();
    let list_top = head.max.y + 2.0;
    let list_h = (p.max.y - list_top).max(0.0);
    let layers: Vec<(LayerId, &cad_doc::Layer)> = session.doc.layers.iter().collect();

    // Scrolling: the wheel over the panel moves the list, clamped so the last
    // row can reach the top but no further. Positive wheel delta scrolls down.
    if ui.hovered(p) && ui.input.scroll.y != 0.0 {
        let max_scroll = (layers.len() as f32 * row - list_h).max(0.0);
        chrome.layer_scroll = chrome_clamp(
            chrome.layer_scroll - ui.input.scroll.y * row * 3.0,
            0.0,
            max_scroll,
        );
    }
    let scroll = chrome.layer_scroll;

    for (i, (id, layer)) in layers.iter().enumerate() {
        let y = list_top + i as f32 * row - scroll;
        // Skip rows entirely above or below the panel. The visible window is
        // bounded, so this keeps the per-frame widget count independent of how
        // many layers the document has.
        if y + row < list_top - 1.0 || y > p.max.y + 1.0 {
            continue;
        }
        let r = Rect::from_xywh(p.min.x + 2.0, y, p.width() - 4.0, row - 2.0);
        let is_current = *id == current;
        if is_current {
            ui.fill_round_rect(r, ui.theme.accent.with_alpha(0.22), ui.theme.border_radius);
        } else if ui.hovered(r) {
            ui.fill_round_rect(r, ui.theme.surface_hover, ui.theme.border_radius);
        }

        // Colour swatch.
        let sw = session.doc.layers.color_of(*id, ui.theme.text);
        let swr = Rect::from_xywh(r.min.x + 4.0, r.center().y - 6.0, 12.0, 12.0);
        ui.fill_round_rect(swr, sw, 2.0);

        // Name.
        let name = layer.name.clone();
        ui.text_sized(
            &name,
            Vec2::new(swr.max.x + 6.0, r.center().y - ui.theme.font_size * 0.5),
            if layer.visible {
                ui.theme.text
            } else {
                ui.theme.text_disabled
            },
            ui.theme.font_size,
        );

        // Visibility and lock, as icons: an eye and a padlock are readable at
        // 20px where "O" and "L" are not, particularly side by side.
        let bw = 20.0;
        let lock_r = Rect::from_xywh(r.max.x - bw - 4.0, r.center().y - 9.0, bw, 18.0);
        let eye_r = Rect::from_xywh(lock_r.min.x - bw - 2.0, lock_r.min.y, bw, 18.0);

        let lock_resp = ui.button(lock_r, true);
        if let Some(ic) = cad_ui::icon("lock") {
            ui.icon(
                ic,
                lock_r,
                if layer.locked || layer.frozen {
                    ui.theme.warning
                } else {
                    ui.theme.text_disabled
                },
                0.62,
            );
        }
        ui.tooltip(lock_r, "Lock");
        let eye_resp = ui.button(eye_r, true);
        if let Some(ic) = cad_ui::icon("eye") {
            ui.icon(
                ic,
                eye_r,
                if layer.visible {
                    ui.theme.accent
                } else {
                    ui.theme.text_disabled
                },
                0.62,
            );
        }
        ui.tooltip(eye_r, if layer.visible { "Hide" } else { "Show" });

        // Clicking the row makes the layer current; the two buttons take priority
        // because they are declared last and would otherwise be shadowed.
        if lock_resp.clicked {
            out.push(Action::ToggleLayerLocked(*id));
        } else if eye_resp.clicked {
            out.push(Action::ToggleLayerVisible(*id));
        } else if ui.hovered(r)
            && !lock_resp.pressed
            && !eye_resp.pressed
            // Only commit on release inside the row, so a drag that starts on a
            // row does not change layers at the end of the drag.
            && ui.input.released.contains(&cad_ui::MouseButton::Left)
        {
            out.push(Action::SetCurrentLayer(*id));
        }
    }
}

/// Clamp helper; `f32::clamp` panics on NaN bounds and this is fed by scroll deltas.
fn chrome_clamp(v: f32, lo: f32, hi: f32) -> f32 {
    if v.is_nan() {
        return lo;
    }
    v.max(lo).min(hi)
}

/// The properties panel: selection details, or document stats when nothing is selected.
fn draw_properties(ui: &mut Ui<'_>, session: &Session, panels: Panels) {
    let p = panels.properties;
    if p.is_empty() {
        return;
    }
    panel_bg(ui, p, ui.theme.background);
    let row = ui.theme.row_height;
    let head = Rect::from_xywh(p.min.x, p.min.y, p.width(), row + 4.0);
    ui.text_sized(
        "PROPERTIES",
        Vec2::new(head.min.x + 8.0, head.min.y + 6.0),
        ui.theme.text_dim,
        ui.theme.font_size,
    );
    ui.fill_rect(
        Rect::from_xywh(p.min.x, head.max.y, p.width(), 1.0),
        ui.theme.border,
    );

    let mut y = head.max.y + 4.0;
    let lx = p.min.x + 8.0;
    let vx = p.min.x + 96.0;

    let field = |ui: &mut Ui<'_>, y: &mut f32, k: &str, v: &str| {
        ui.text_sized(k, Vec2::new(lx, *y), ui.theme.text_dim, ui.theme.font_size);
        let w = (p.max.x - vx - 8.0).max(0.0);
        ui.text_sized(v, Vec2::new(vx, *y), ui.theme.text, ui.theme.font_size);
        let _ = w;
        *y += row;
    };

    let (planar, solid) = session.entity_summary();
    field(ui, &mut y, "Drawing", session.doc.name());
    field(ui, &mut y, "Entities", &format!("{planar} 2D / {solid} 3D"));
    field(
        ui,
        &mut y,
        "Layers",
        &format!("{}", session.doc.layers.len()),
    );
    field(ui, &mut y, "Units", session.doc.units.label());
    field(
        ui,
        &mut y,
        "Undo",
        &format!("{} steps", session.history.undo_depth()),
    );
    field(
        ui,
        &mut y,
        "View",
        match session.viewport.mode {
            ViewMode::Model2d => "2D drafting",
            ViewMode::Model3d => "3D model",
        },
    );
    field(
        ui,
        &mut y,
        "Grid",
        &format!("{:.3}", session.viewport.grid_spacing),
    );

    y += 4.0;
    ui.fill_rect(
        Rect::from_xywh(p.min.x + 6.0, y, p.width() - 12.0, 1.0),
        ui.theme.border,
    );
    y += 6.0;

    let sel = &session.tool.selection;
    if sel.is_empty() {
        ui.text_sized(
            "Nothing selected",
            Vec2::new(lx, y),
            ui.theme.text_dim,
            ui.theme.font_size,
        );
        return;
    }

    let first = sel.first().and_then(|i| session.doc.entities.get(*i));
    let Some(e) = first else {
        ui.text_sized(
            "Selection is stale",
            Vec2::new(lx, y),
            ui.theme.warning,
            ui.theme.font_size,
        );
        return;
    };

    field(ui, &mut y, "Selected", &format!("{}", sel.len()));
    field(ui, &mut y, "Type", e.type_name());
    field(ui, &mut y, "Layer", session.doc.layers.name(e.layer()));
    if let Some((v, is_area)) = e.measure() {
        let unit = session.doc.units.suffix();
        field(
            ui,
            &mut y,
            if is_area { "Area" } else { "Length" },
            &format!("{v:.3}{unit}"),
        );
    }
    match &e.entity {
        EntityKind::Line(l) => {
            field(
                ui,
                &mut y,
                "Start",
                &format!("{:.3}, {:.3}", l.p0.x, l.p0.y),
            );
            field(ui, &mut y, "End", &format!("{:.3}, {:.3}", l.p1.x, l.p1.y));
            // No `angle_deg` on `Line`: derive it from the direction so this does
            // not depend on a method that does not exist.
            let d = l.p1 - l.p0;
            field(
                ui,
                &mut y,
                "Angle",
                &format!("{:.2}deg", d.y.atan2(d.x).to_degrees()),
            );
        }
        EntityKind::Circle(c) => {
            field(
                ui,
                &mut y,
                "Centre",
                &format!("{:.3}, {:.3}", c.center.x, c.center.y),
            );
            field(ui, &mut y, "Radius", &format!("{:.3}", c.radius));
        }
        EntityKind::Box(b) => {
            // `Vec3` has no `Display`; print the components.
            field(
                ui,
                &mut y,
                "Min",
                &format!("{:.3}, {:.3}, {:.3}", b.min.x, b.min.y, b.min.z),
            );
            field(
                ui,
                &mut y,
                "Max",
                &format!("{:.3}, {:.3}, {:.3}", b.max.x, b.max.y, b.max.z),
            );
            field(ui, &mut y, "Volume", &format!("{:.3}", b.volume()));
        }
        EntityKind::Mesh(m) => {
            field(ui, &mut y, "Triangles", &m.triangle_count().to_string());
            let b = m.bounds();
            field(
                ui,
                &mut y,
                "Size",
                &format!(
                    "{:.2} x {:.2} x {:.2}",
                    b.max.x - b.min.x,
                    b.max.y - b.min.y,
                    b.max.z - b.min.z
                ),
            );
        }
        _ => {}
    }
    field(ui, &mut y, "Colour", &format!("ACI {}", e.common.color));
}

/// The command line: a real text field, model-driven completion, and submit.
fn draw_command_line(
    ui: &mut Ui<'_>,
    chrome: &mut Chrome,
    session: &Session,
    panels: Panels,
    suggestions: &Suggestions,
    out: &mut Vec<Action>,
) {
    let p = panels.command_line;
    if p.is_empty() {
        return;
    }
    panel_bg(ui, p, ui.theme.surface);

    let prompt = session
        .tool
        .prompt()
        .unwrap_or_else(|| "Command".to_string());
    let pw = text_width(&prompt, ui.theme.font_size) + 8.0;
    let y = p.center().y - ui.theme.font_size * 0.5;
    // A prompt badge rather than bare text: while a tool is mid-sequence the
    // prompt is the most important thing on the screen.
    let plate = Rect::from_xywh(p.min.x + 6.0, p.min.y + 2.0, pw + 6.0, p.height() - 4.0);
    if session.tool.prompt().is_some() {
        ui.fill_round_rect(
            plate,
            ui.theme.accent.with_alpha(0.18),
            ui.theme.border_radius,
        );
        ui.stroke_rect(plate, ui.theme.accent.with_alpha(0.45), 1.0);
    }
    ui.text_sized(
        &prompt,
        Vec2::new(p.min.x + 14.0, y),
        if session.tool.prompt().is_some() {
            ui.theme.accent
        } else {
            ui.theme.text_dim
        },
        ui.theme.font_size,
    );

    let field = Rect::from_xywh(
        p.min.x + pw + 12.0,
        p.min.y + 2.0,
        (p.width() - pw - 20.0).max(20.0),
        (p.height() - 4.0).max(8.0),
    );
    let (_resp, _changed) = ui.text_field(field, &mut chrome.command, true);

    // Ghost text: the tail of the top completion, drawn under what was typed so
    // the user can see the whole word without leaving the caret.
    if !chrome.command.focused {
        // Nothing typed and nothing focused: offer the model's next guess.
        if let Some(next) = suggestions.next
            && chrome.command.text.is_empty()
        {
            let hint = format!("{next}  (predicted)");
            ui.text_sized(
                &hint,
                Vec2::new(field.min.x + 6.0, y),
                ui.theme.text_disabled,
                ui.theme.font_size,
            );
        }
    }
    if let Some(ghost) = &suggestions.ghost
        && !ghost.is_empty()
    {
        let typed_w = text_width(&chrome.command.text, ui.theme.font_size);
        ui.text_sized(
            ghost,
            Vec2::new(field.min.x + 6.0 + typed_w, y),
            ui.theme.text_disabled,
            ui.theme.font_size,
        );
    }

    // Tab accepts the top completion. The text field does not consume Tab, so
    // this cannot double-fire.
    if ui.input.key_pressed(cad_ui::input::Key::Tab)
        && let Some(top) = suggestions.completions.first()
    {
        out.push(Action::AcceptCompletion(top.clone()));
    }

    // Enter is consumed by the field (it drops focus), so submission has to be
    // detected from the same key press the field acted on.
    if ui.input.key_pressed(cad_ui::input::Key::Enter) && !chrome.command.text.is_empty() {
        out.push(Action::Submit(chrome.command.text.clone()));
    }
}

/// The status bar: coordinates, zoom, counts, toggles and the adapter name.
fn draw_status_bar(
    ui: &mut Ui<'_>,
    session: &Session,
    panels: Panels,
    facts: StatusFacts,
    _suggestions: &Suggestions,
    out: &mut Vec<Action>,
) {
    let p = panels.status_bar;
    if p.is_empty() {
        return;
    }
    panel_bg(ui, p, ui.theme.surface);
    ui.fill_rect(
        Rect::from_xywh(p.min.x, p.min.y, p.width(), 1.0),
        ui.theme.border,
    );

    let f = ui.theme.font_size;
    let y = p.center().y - f * 0.5;
    let mut x = p.min.x + 8.0;
    let seg = ui.theme.text_dim;
    let strong = ui.theme.text;

    let put = |ui: &mut Ui<'_>, x: &mut f32, label: &str, color: Rgba| {
        if *x + text_width(label, f) + 14.0 > p.max.x {
            return;
        }
        ui.text_sized(label, Vec2::new(*x, y), color, f);
        *x += text_width(label, f) + 14.0;
    };

    put(ui, &mut x, &format!("X {:.3}", facts.cursor.x), strong);
    put(ui, &mut x, &format!("Y {:.3}", facts.cursor.y), strong);
    put(ui, &mut x, &format!("Z {}", facts.scale), seg);
    put(ui, &mut x, session.tool.id.label(), strong);
    put(ui, &mut x, &format!("{} ents", facts.entities), seg);
    if facts.selection > 0 {
        put(
            ui,
            &mut x,
            &format!("{} sel", facts.selection),
            ui.theme.accent,
        );
    }

    // Toggles, right-aligned so they do not move when the read-outs change.
    // One widget decides on/off in one place, which is what stops the four from
    // drifting into four different greys.
    let toggles: [(&str, bool); 4] = [
        ("GRID", session.viewport.show_grid),
        ("OSNAP", session.snap.settings.enabled),
        ("ORTHO", session.snap.settings.has(SnapKind::Ortho)),
        ("POLAR", session.snap.settings.has(SnapKind::Polar)),
    ];
    let mut rx = p.max.x - 8.0;
    for (label, on) in toggles.iter().rev() {
        let w = text_width(label, f) + 12.0;
        let r = Rect::from_xywh(rx - w, p.min.y + 3.0, w, p.height() - 6.0);
        // The click *is* the toggle, so a status bar is a control and not only
        // a read-out. `chrome::apply_action` runs the matching command.
        let (resp, _color) = ui.chip(r, label, *on);
        if resp.clicked {
            out.push(Action::Command((*label).to_lowercase()));
        }
        rx -= w + 4.0;
    }

    let info = format!("{:.0} fps  {}", facts.fps, facts.adapter);
    let iw = text_width(&info, f);
    if rx - iw - 10.0 > x {
        ui.text_sized(&info, Vec2::new(rx - iw, y), ui.theme.text_disabled, f);
    }
}

/// Apply one action to the session.
///
/// This is the only place that writes, which is why the widget pass can take
/// `&Session`: a click cannot mutate a document it is in the middle of drawing.
pub fn apply_action(action: &Action, session: &mut Session, chrome: &mut Chrome) {
    match action {
        Action::Command(name) => {
            let _ = session.activate(name);
        }
        Action::Submit(line) => {
            chrome.command.text.clear();
            chrome.command.cursor = 0;
            chrome.command.sel_start = None;
            session.run_command_line(line);
        }
        Action::AcceptCompletion(word) => {
            chrome.command.text = word.clone();
            chrome.command.cursor = word.chars().count();
            chrome.command.sel_start = None;
        }
        Action::ToggleLayerVisible(id) => session.toggle_layer_visibility(*id),
        Action::ToggleLayerLocked(id) => session.toggle_layer_lock(*id),
        Action::SetCurrentLayer(id) => session.set_current_layer(*id),
        Action::NewLayer => session.add_layer(""),
        Action::NewDocument => session.new_document(),
        Action::Open => {
            session.status = StatusMessage::prompt("Type OPEN <path> at the command line");
        }
        Action::Save => {
            session.save(None);
        }
        Action::SaveAs => {
            session.status = StatusMessage::prompt("Type SAVEAS <path> at the command line");
        }
        Action::Export => {
            session.status = StatusMessage::prompt("Type EXPORT <path>.dxf at the command line");
        }
        Action::About => {
            session.status = StatusMessage::info(concat!(
                "CADKit - native 2D/3D CAD on wgpu. ",
                "49 commands, DXF and native formats, analytic geometry."
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panels_tile_the_window() {
        let p = Panels::layout(1600.0, 900.0);
        assert_eq!(p.menu_bar.max.y, p.ribbon.min.y);
        assert_eq!(p.ribbon.max.y, p.canvas.min.y);
        assert_eq!(p.canvas.min.x, p.layer_panel.max.x);
        assert_eq!(p.canvas.max.x, p.properties.min.x);
        assert_eq!(p.canvas.max.y, p.command_line.min.y);
        assert_eq!(p.command_line.max.y, p.status_bar.min.y);
        assert!((p.status_bar.max.y - 900.0).abs() < 1e-3);
    }

    #[test]
    fn canvas_never_covers_a_panel() {
        let p = Panels::layout(1200.0, 800.0);
        for panel in [
            p.layer_panel,
            p.properties,
            p.ribbon,
            p.menu_bar,
            p.command_line,
            p.status_bar,
        ] {
            assert!(
                !p.canvas.intersect(panel).has_area(),
                "canvas and {panel:?} share area"
            );
        }
    }

    #[test]
    fn panels_survive_a_tiny_window() {
        for (w, h) in [(100.0f32, 50.0f32), (10.0, 10.0), (0.0, 0.0)] {
            let p = Panels::layout(w, h);
            assert!(p.canvas.width() >= 0.0, "{w}x{h}");
            assert!(p.canvas.height() >= 0.0, "{w}x{h}");
            assert!(p.properties.min.x >= 0.0, "{w}x{h}");
        }
    }

    #[test]
    fn ribbon_tabs_are_inside_the_ribbon() {
        let p = Panels::layout(1600.0, 900.0);
        let t = cad_ui::Theme::dark();
        let tabs = ribbon_tabs_rects(p.ribbon, t);
        assert_eq!(tabs.len(), RibbonTab::ALL.len());
        for r in &tabs {
            assert!(p.ribbon.contains(r.min), "{r:?} escapes the ribbon");
            assert!(p.ribbon.contains(r.max), "{r:?} escapes the ribbon");
        }
        // Tabs must not overlap each other or only the leftmost would be clickable.
        for w in tabs.windows(2) {
            assert!(w[0].max.x <= w[1].min.x + 1e-3, "{:?}", tabs);
        }
    }

    #[test]
    fn every_ribbon_button_is_a_real_command() {
        // A button that runs nothing is worse than no button.
        let reg = cad_app::command::CommandRegistry::new();
        for tab in RibbonTab::ALL {
            for g in tab.groups() {
                for (icon_name, cmd, label, _) in g.buttons {
                    assert!(
                        reg.get(cmd).is_some(),
                        "{cmd} (tab {}) is not a command",
                        tab.label()
                    );
                    assert!(!label.is_empty(), "{cmd} has no caption");
                    // A missing icon renders an empty plate, which reads as a
                    // broken button rather than an unlabelled one.
                    assert!(
                        cad_ui::icons::by_name(icon_name).is_some(),
                        "{cmd} names non-existent icon {icon_name}"
                    );
                }
            }
        }
    }

    #[test]
    fn every_menu_has_entries() {
        let reg = cad_app::command::CommandRegistry::new();
        for m in MENUS {
            assert!(!reg.menu_entries(m).is_empty(), "{m} is empty");
        }
    }

    #[test]
    fn title_marks_unsaved_changes() {
        // A fresh session starts dirty (nothing has been saved yet), so this has
        // to clear the flag to see the marker appear.
        let mut s = Session::new();
        assert!(s.dirty, "a new drawing has unsaved state");
        assert!(title_of(&s).contains('*'), "{}", title_of(&s));
        s.dirty = false;
        assert!(!title_of(&s).contains('*'), "{}", title_of(&s));
    }

    #[test]
    fn title_uses_the_file_name_when_there_is_one() {
        let mut s = Session::new();
        s.dirty = false;
        s.path = Some(std::path::PathBuf::from("/tmp/plan.dxf"));
        assert_eq!(title_of(&s), "plan.dxf", "{}", title_of(&s));
        s.path = None;
        assert_eq!(title_of(&s), "Untitled");
    }

    #[test]
    fn clamp_handles_nan() {
        assert_eq!(chrome_clamp(f32::NAN, 0.0, 10.0), 0.0);
        assert_eq!(chrome_clamp(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(chrome_clamp(50.0, 0.0, 10.0), 10.0);
        assert_eq!(chrome_clamp(5.0, 0.0, 10.0), 5.0);
    }

    #[test]
    fn ghost_text_is_only_the_missing_tail() {
        let s = Suggestions::build("li", vec!["line".into(), "list".into()], None);
        assert_eq!(s.ghost.as_deref(), Some("ne"));
        assert!(s.has());
    }

    #[test]
    fn a_completion_that_does_not_extend_the_prefix_has_no_ghost() {
        // `rank` also returns near-misses that are not prefix extensions; those
        // must not be drawn inline, because the ghost would not line up with
        // what the user typed.
        let s = Suggestions::build("ln", vec!["line".into()], None);
        assert!(s.ghost.is_none(), "{s:?}");
    }

    #[test]
    fn an_empty_prefix_has_no_completions() {
        let s = Suggestions::build("", Vec::new(), Some("line"));
        assert!(!s.has());
        assert!(s.ghost.is_none());
        assert_eq!(s.next, Some("line"));
    }

    #[test]
    fn suggestions_default_to_nothing() {
        let s = Suggestions::default();
        assert!(!s.has());
        assert!(s.ghost.is_none());
        assert!(s.next.is_none());
    }

    #[test]
    fn accepting_a_completion_replaces_the_whole_line() {
        let mut s = Session::new();
        let mut c = Chrome::new();
        apply_action(
            &Action::AcceptCompletion("rectangle".into()),
            &mut s,
            &mut c,
        );
        assert_eq!(c.command.text, "rectangle");
        assert_eq!(c.command.cursor, "rectangle".chars().count());
        // Accepting is not submitting: nothing ran.
        assert_eq!(s.tool.id, cad_app::tools::ToolId::Select);
    }

    #[test]
    fn submitting_clears_the_line_after_running() {
        let mut s = Session::new();
        let mut c = Chrome::new();
        c.command.text = "line".into();
        apply_action(&Action::Submit("line".into()), &mut s, &mut c);
        assert_eq!(
            c.command.text, "",
            "the line must be cleared after submitting"
        );
        assert_eq!(s.tool.id, cad_app::tools::ToolId::Line);
        assert_eq!(s.recent_commands(1), vec!["line"]);
    }

    #[test]
    fn rect_contains_its_own_corners() {
        // The layout relies on this for tab and button hit-testing.
        let r = Rect::from_xywh(10.0, 10.0, 20.0, 20.0);
        assert!(r.contains(r.min));
        assert!(r.contains(r.max));
    }
}
