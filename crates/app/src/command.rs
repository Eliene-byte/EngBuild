//! The command registry: named, discoverable actions with keyboard shortcuts.
//!
//! This is what makes the app scriptable and what the command line parses. It
//! is deliberately data-driven so the UI (menu bar, command line, help) and the
//! keyboard all dispatch through exactly one table.

use std::collections::BTreeMap;

/// Result of running a command.
#[derive(Debug, Clone, PartialEq)]
pub enum CommandResult {
    /// Done.
    Ok,
    /// Not applicable right now (the menu item should be greyed out).
    Unavailable,
    /// Failed with a message for the command line.
    Error(String),
    /// The command started an interactive sequence.
    Started,
}

impl CommandResult {
    pub fn is_ok(&self) -> bool {
        matches!(self, CommandResult::Ok | CommandResult::Started)
    }
    /// Text to show on the command line, if any.
    pub fn message(&self) -> Option<&str> {
        match self {
            CommandResult::Error(m) => Some(m.as_str()),
            _ => None,
        }
    }
}

/// One entry in the command table.
#[derive(Debug, Clone)]
pub struct Command {
    pub name: &'static str,
    /// Localized-ish label for menus.
    pub label: &'static str,
    /// Shortcut in the form `Ctrl+S`, `F2`, `Del`.
    pub shortcut: Option<&'static str>,
    pub category: &'static str,
    /// Menu path, e.g. `["Edit", "Undo"]`.
    pub menu: &'static [&'static str],
    /// Documentation shown by `?`.
    pub help: &'static str,
}

impl Command {
    pub fn new(
        name: &'static str,
        label: &'static str,
        category: &'static str,
        help: &'static str,
    ) -> Self {
        Self {
            name,
            label,
            category,
            help,
            shortcut: None,
            menu: &[],
        }
    }
    pub fn shortcut(mut self, s: &'static str) -> Self {
        self.shortcut = Some(s);
        self
    }
    pub fn menu(mut self, path: &'static [&'static str]) -> Self {
        self.menu = path;
        self
    }
}

/// All commands, indexed by name and by shortcut.
#[derive(Debug, Default)]
pub struct CommandRegistry {
    commands: Vec<Command>,
    by_name: BTreeMap<&'static str, usize>,
    by_shortcut: BTreeMap<String, usize>,
}

impl CommandRegistry {
    pub fn new() -> Self {
        let mut r = Self::default();
        r.add_builtins();
        r.reindex();
        r
    }

    /// Empty registry (tests build their own).
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn register(&mut self, c: Command) -> usize {
        if let Some(i) = self.by_name.get(c.name) {
            self.commands[*i] = c;
            return *i;
        }
        let i = self.commands.len();
        self.by_name.insert(c.name, i);
        if let Some(s) = c.shortcut {
            self.by_shortcut.insert(s.to_ascii_lowercase(), i);
        }
        self.commands.push(c);
        i
    }

    fn reindex(&mut self) {
        self.by_name.clear();
        self.by_shortcut.clear();
        for (i, c) in self.commands.iter().enumerate() {
            self.by_name.insert(c.name, i);
            if let Some(s) = c.shortcut {
                self.by_shortcut.insert(s.to_ascii_lowercase(), i);
            }
        }
    }

    pub fn get(&self, name: &str) -> Option<&Command> {
        self.by_name
            .get(name.to_ascii_lowercase().as_str())
            .map(|i| &self.commands[*i])
    }
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.by_name
            .get(name.to_ascii_lowercase().as_str())
            .copied()
    }
    pub fn at(&self, i: usize) -> Option<&Command> {
        self.commands.get(i)
    }
    pub fn by_shortcut(&self, shortcut: &str) -> Option<&Command> {
        self.by_shortcut
            .get(&shortcut.to_ascii_lowercase())
            .map(|i| &self.commands[*i])
    }
    pub fn len(&self) -> usize {
        self.commands.len()
    }
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = &Command> {
        self.commands.iter()
    }
    /// All commands in a menu, as `(menu path, index)` pairs, in declaration
    /// order — which is the order the menu bar shows them.
    pub fn menu_entries(&self, top: &str) -> Vec<usize> {
        self.commands
            .iter()
            .enumerate()
            .filter(|(_, c)| c.menu.first() == Some(&top))
            .map(|(i, _)| i)
            .collect()
    }
    /// Search by substring, for the command line's autocomplete.
    pub fn search(&self, prefix: &str) -> Vec<&Command> {
        let p = prefix.to_ascii_lowercase();
        self.commands
            .iter()
            .filter(|c| c.name.to_ascii_lowercase().starts_with(&p))
            .collect()
    }

    fn add_builtins(&mut self) {
        // File
        self.register(
            Command::new("new", "New", "File", "Start a new drawing.")
                .shortcut("Ctrl+N")
                .menu(&["File", "New"]),
        );
        self.register(
            Command::new("open", "Open...", "File", "Open an existing drawing.")
                .shortcut("Ctrl+O")
                .menu(&["File", "Open"]),
        );
        self.register(
            Command::new("save", "Save", "File", "Save the current drawing.")
                .shortcut("Ctrl+S")
                .menu(&["File", "Save"]),
        );
        self.register(
            Command::new(
                "saveas",
                "Save As...",
                "File",
                "Save the drawing under a new name.",
            )
            .shortcut("Ctrl+Shift+S")
            .menu(&["File", "Save As"]),
        );
        self.register(
            Command::new(
                "export",
                "Export...",
                "File",
                "Write a DXF of the current drawing.",
            )
            .menu(&["File", "Export"]),
        );

        // Edit
        self.register(
            Command::new("undo", "Undo", "Edit", "Undo the last operation.")
                .shortcut("Ctrl+Z")
                .menu(&["Edit", "Undo"]),
        );
        self.register(
            Command::new("redo", "Redo", "Edit", "Redo the last undone operation.")
                .shortcut("Ctrl+Y")
                .menu(&["Edit", "Redo"]),
        );
        self.register(
            Command::new("cut", "Cut to Clipboard", "Edit", "Cut the selection.")
                .shortcut("Ctrl+X")
                .menu(&["Edit", "Cut"]),
        );
        self.register(
            Command::new("copy", "Copy to Clipboard", "Edit", "Copy the selection.")
                .shortcut("Ctrl+C")
                .menu(&["Edit", "Copy"]),
        );
        self.register(
            Command::new("paste", "Paste", "Edit", "Paste from the clipboard.")
                .shortcut("Ctrl+V")
                .menu(&["Edit", "Paste"]),
        );

        // View
        self.register(
            Command::new(
                "zoomall",
                "Zoom Extents",
                "View",
                "Fit the drawing in the viewport.",
            )
            .shortcut("Z")
            .menu(&["View", "Zoom Extents"]),
        );
        self.register(
            Command::new(
                "zoomwindow",
                "Zoom Window",
                "View",
                "Zoom to a dragged rectangle.",
            )
            .menu(&["View", "Zoom Window"]),
        );
        self.register(
            Command::new("zoomin", "Zoom In", "View", "Zoom in one step.")
                .menu(&["View", "Zoom In"]),
        );
        self.register(
            Command::new("zoomout", "Zoom Out", "View", "Zoom out one step.")
                .menu(&["View", "Zoom Out"]),
        );
        self.register(
            Command::new(
                "view3d",
                "3D Orbit",
                "View",
                "Switch to the model-space 3D view.",
            )
            .menu(&["View", "3D Orbit"]),
        );
        self.register(
            Command::new("grid", "Grid", "View", "Toggle the drawing grid.").shortcut("F7"),
        );
        self.register(
            Command::new("snap", "Object Snap", "Tools", "Toggle object snapping.")
                .shortcut("F3")
                .menu(&["Tools", "Object Snap"]),
        );
        self.register(
            Command::new(
                "ortho",
                "Ortho Mode",
                "Tools",
                "Constrain input to the axes.",
            )
            .shortcut("F8")
            .menu(&["Tools", "Ortho Mode"]),
        );
        self.register(
            Command::new(
                "polar",
                "Polar Tracking",
                "Tools",
                "Constrain input to increments.",
            )
            .shortcut("F10")
            .menu(&["Tools", "Polar Tracking"]),
        );

        // Draw
        self.register(
            Command::new("line", "Line", "Draw", "Draw a straight line.").menu(&["Draw", "Line"]),
        );
        self.register(
            Command::new(
                "circle",
                "Circle",
                "Draw",
                "Draw a circle by centre and radius.",
            )
            .menu(&["Draw", "Circle"]),
        );
        self.register(
            Command::new("arc", "Arc", "Draw", "Draw an arc by three points.")
                .menu(&["Draw", "Arc"]),
        );
        self.register(
            Command::new("polyline", "Polyline", "Draw", "Draw a connected polyline.")
                .menu(&["Draw", "Polyline"]),
        );
        self.register(
            Command::new(
                "rectangle",
                "Rectangle",
                "Draw",
                "Draw an axis-aligned rectangle.",
            )
            .menu(&["Draw", "Rectangle"]),
        );
        self.register(
            Command::new("spline", "Spline", "Draw", "Draw a smooth curve.")
                .menu(&["Draw", "Spline"]),
        );
        self.register(
            Command::new("text", "Text", "Draw", "Place a single-line text.")
                .menu(&["Draw", "Text"]),
        );

        // Modify
        self.register(
            Command::new("erase", "Erase", "Modify", "Delete the selected entities.")
                .menu(&["Modify", "Erase"]),
        );
        self.register(
            Command::new("move", "Move", "Modify", "Move the selection.").menu(&["Modify", "Move"]),
        );
        self.register(
            Command::new("copytool", "Copy", "Modify", "Copy the selection.")
                .menu(&["Modify", "Copy"]),
        );
        self.register(
            Command::new("rotate", "Rotate", "Modify", "Rotate the selection.")
                .menu(&["Modify", "Rotate"]),
        );
        self.register(
            Command::new("scale", "Scale", "Modify", "Scale the selection.")
                .menu(&["Modify", "Scale"]),
        );
        self.register(
            Command::new("offset", "Offset", "Modify", "Offset curves at a distance.")
                .menu(&["Modify", "Offset"]),
        );
        self.register(
            Command::new("trim", "Trim", "Modify", "Trim curves to a boundary.")
                .menu(&["Modify", "Trim"]),
        );
        self.register(
            Command::new("extend", "Extend", "Modify", "Extend curves to a boundary.")
                .menu(&["Modify", "Extend"]),
        );
        self.register(
            Command::new(
                "mirror",
                "Mirror",
                "Modify",
                "Mirror the selection across an axis.",
            )
            .menu(&["Modify", "Mirror"]),
        );
        self.register(
            Command::new(
                "array",
                "Array",
                "Modify",
                "Create a rectangular array of the selection.",
            )
            .menu(&["Modify", "Array"]),
        );

        // 3D
        self.register(Command::new("box", "Box", "3D", "Create a 3D box.").menu(&["3D", "Box"]));
        self.register(
            Command::new("extrude", "Extrude", "3D", "Give a closed region a height.")
                .menu(&["3D", "Extrude"]),
        );

        // Help
        self.register(
            Command::new("help", "Help", "Help", "List every command.")
                .shortcut("F1")
                .menu(&["Help", "Help"]),
        );
        self.register(
            Command::new("about", "About", "Help", "Version and licence information.")
                .menu(&["Help", "About"]),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_are_registered_with_unique_names() {
        let r = CommandRegistry::new();
        assert!(r.len() > 40, "only {} commands", r.len());
        let mut seen = std::collections::HashSet::new();
        for c in r.iter() {
            assert!(seen.insert(c.name), "duplicate command name {}", c.name);
        }
    }

    #[test]
    fn lookup_is_case_insensitive() {
        let r = CommandRegistry::new();
        assert_eq!(r.get("LINE").map(|c| c.name), Some("line"));
        assert_eq!(r.get("Line").map(|c| c.name), Some("line"));
        assert!(r.get("nope").is_none());
    }

    #[test]
    fn shortcuts_are_unique_and_resolvable() {
        let r = CommandRegistry::new();
        let mut seen = std::collections::HashSet::new();
        for c in r.iter() {
            if let Some(s) = c.shortcut {
                assert!(seen.insert(s), "duplicate shortcut {s}");
                assert_eq!(r.by_shortcut(s).map(|x| x.name), Some(c.name));
                assert_eq!(
                    r.by_shortcut(&s.to_uppercase()).map(|x| x.name),
                    Some(c.name)
                );
            }
        }
    }

    #[test]
    fn every_command_has_help_and_a_label() {
        let r = CommandRegistry::new();
        for c in r.iter() {
            assert!(!c.label.is_empty(), "{} has no label", c.name);
            assert!(!c.help.is_empty(), "{} has no help", c.name);
            assert!(!c.category.is_empty(), "{} has no category", c.name);
        }
    }

    #[test]
    fn menu_paths_are_grouped() {
        let r = CommandRegistry::new();
        for top in [
            "File", "Edit", "View", "Tools", "Draw", "Modify", "3D", "Help",
        ] {
            assert!(!r.menu_entries(top).is_empty(), "{top} menu is empty");
        }
        // Declaration order is the display order.
        let file = r.menu_entries("File");
        let names: Vec<&str> = file
            .iter()
            .filter_map(|i| r.at(*i))
            .map(|c| c.name)
            .collect();
        assert_eq!(names[0], "new");
        assert_eq!(names[1], "open");
    }

    #[test]
    fn autocomplete_prefix_search() {
        let r = CommandRegistry::new();
        let hits = r.search("zoo");
        assert!(hits.iter().any(|c| c.name == "zoomall"));
        assert!(!hits.iter().any(|c| c.name == "line"));
        assert!(r.search("zzzz").is_empty());
    }

    #[test]
    fn registering_twice_replaces() {
        let mut r = CommandRegistry::empty();
        r.register(Command::new("a", "A", "T", "first"));
        r.register(Command::new("a", "A2", "T", "second"));
        assert_eq!(r.len(), 1);
        assert_eq!(r.get("a").unwrap().label, "A2");
        assert_eq!(r.get("a").unwrap().help, "second");
    }

    #[test]
    fn results_carry_messages() {
        assert!(CommandResult::Ok.is_ok());
        assert!(CommandResult::Started.is_ok());
        assert!(!CommandResult::Unavailable.is_ok());
        let e = CommandResult::Error("nope".into());
        assert!(!e.is_ok());
        assert_eq!(e.message(), Some("nope"));
        assert_eq!(CommandResult::Ok.message(), None);
    }

    #[test]
    fn empty_registry_has_nothing() {
        let r = CommandRegistry::empty();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.iter().next().is_none());
    }
}
