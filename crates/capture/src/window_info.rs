//! Foreground-window inspection — what app is the user looking at?
//!
//! This is the third detection axis (after OCR-text patterns and QR
//! decoding). It answers a different question: not "what's on screen?"
//! but "which app's window is in front?" If the answer is `Phantom.exe`
//! or `Tonkeeper Desktop` or "Notepad — secrets.env", we don't need
//! to OCR a single pixel: the entire window region is sensitive.
//!
//! Implementation:
//!   - **Windows**: `GetForegroundWindow` + `GetWindowTextW` +
//!     `GetWindowRect` + `GetWindowThreadProcessId` +
//!     `QueryFullProcessImageNameW`. All public Win32 APIs.
//!   - **macOS** / **Linux**: stubs for now. The same concept maps to
//!     `NSWorkspace.frontmostApplication` (macOS) and
//!     `xprop -root _NET_ACTIVE_WINDOW` / `gdbus` (Linux).
//!
//! The pattern list `SENSITIVE_PATTERNS` is intentionally tight —
//! adding a new app is one line and a re-release.

use crate::CaptureError;

/// Information about the currently-foreground window.
#[derive(Debug, Clone)]
pub struct WindowInfo {
    /// Window title, as the OS reports it.
    pub title: String,
    /// Process executable name without path (`Phantom.exe`, `Code.exe`).
    pub process_name: String,
    /// Bounding rectangle of the window in screen-pixel coordinates.
    pub rect: WindowRect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowRect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensitivityVerdict {
    /// Nothing matched — leave the window alone, let OCR / QR do their thing.
    NotSensitive,
    /// Matched a known crypto wallet — blur the entire window rect.
    CryptoWallet,
    /// Matched an editor with what looks like a secrets file open
    /// (filename contains `.env`, `secrets`, `credentials`, etc.).
    SecretsEditor,
    /// Matched a terminal with shell prompts visible — they typically
    /// echo just-pasted credentials.
    Terminal,
}

impl SensitivityVerdict {
    #[must_use]
    pub fn is_sensitive(self) -> bool {
        !matches!(self, Self::NotSensitive)
    }
}

/// Decide if a `WindowInfo` should be treated as sensitive.
#[must_use]
pub fn classify(info: &WindowInfo) -> SensitivityVerdict {
    let title = info.title.to_lowercase();
    let proc = info.process_name.to_lowercase();

    for pat in CRYPTO_WALLET_PATTERNS {
        if proc.contains(pat) || title.contains(pat) {
            return SensitivityVerdict::CryptoWallet;
        }
    }

    // Editors get classified as "SecretsEditor" only when the title
    // *also* mentions a secret-like filename. The editor process
    // alone isn't sensitive (people use editors for everything).
    for editor in EDITOR_PROCESSES {
        if proc.contains(editor) {
            for hint in SECRET_FILENAME_HINTS {
                if title.contains(hint) {
                    return SensitivityVerdict::SecretsEditor;
                }
            }
        }
    }

    for term in TERMINAL_PROCESSES {
        if proc.contains(term) {
            return SensitivityVerdict::Terminal;
        }
    }

    SensitivityVerdict::NotSensitive
}

// ─── Pattern lists ──────────────────────────────────────────────────

/// Process names / window titles that ARE a crypto wallet (any match → blur).
const CRYPTO_WALLET_PATTERNS: &[&str] = &[
    // Browser-extension wallets (popup window names usually include
    // the brand)
    "phantom",
    "metamask",
    "trust wallet",
    "trustwallet",
    "tonkeeper",
    "rabby",
    "rainbow",
    "coinbase wallet",
    "backpack",
    "solflare",
    "exodus",
    "wasabi",
    "electrum",
    "ledger live",
    "trezor suite",
];

/// Editors whose window title we inspect for secret-file hints.
const EDITOR_PROCESSES: &[&str] = &[
    "notepad.exe",
    "notepad++.exe",
    "code.exe",         // VS Code
    "rider64.exe",
    "idea64.exe",
    "pycharm64.exe",
    "sublime_text.exe",
    "subl.exe",
    "atom.exe",
    "vim.exe",
    "nvim.exe",
];

/// If an editor's title contains any of these, the file open in it
/// is treated as sensitive.
const SECRET_FILENAME_HINTS: &[&str] = &[
    ".env",
    "secrets",
    "credentials",
    "private",
    ".pem",
    ".key",
    "id_rsa",
    "id_ed25519",
    "wallet.dat",
    "keystore",
    "mnemonic",
    "seed.txt",
];

/// Terminals — always sensitive. People paste keys, the screen scrolls,
/// and the keys stay visible in the scrollback.
const TERMINAL_PROCESSES: &[&str] = &[
    "windowsterminal.exe",
    "wt.exe",
    "powershell.exe",
    "pwsh.exe",
    "cmd.exe",
    "conhost.exe",
    "alacritty.exe",
    "wezterm-gui.exe",
];

// ─── Per-OS bindings ────────────────────────────────────────────────

#[cfg(target_os = "windows")]
pub fn get_foreground_window() -> Result<Option<WindowInfo>, CaptureError> {
    use ::windows::Win32::Foundation::{CloseHandle, RECT};
    use ::windows::Win32::System::ProcessStatus::GetModuleBaseNameW;
    use ::windows::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ};
    use ::windows::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowRect, GetWindowTextLengthW, GetWindowTextW,
        GetWindowThreadProcessId,
    };

    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.0.is_null() {
            return Ok(None);
        }

        // Title.
        let len = GetWindowTextLengthW(hwnd);
        let mut title_buf = vec![0u16; (len as usize).max(1) + 1];
        let copied = GetWindowTextW(hwnd, &mut title_buf);
        title_buf.truncate(copied as usize);
        let title = String::from_utf16_lossy(&title_buf);

        // Rectangle.
        let mut rect = RECT::default();
        let _ = GetWindowRect(hwnd, &mut rect);
        let rect = WindowRect {
            x: rect.left,
            y: rect.top,
            w: rect.right - rect.left,
            h: rect.bottom - rect.top,
        };

        // Process name.
        let mut pid: u32 = 0;
        let _ = GetWindowThreadProcessId(hwnd, Some(&mut pid));
        let process_name = if pid != 0 {
            match OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,
                false,
                pid,
            ) {
                Ok(handle) if !handle.0.is_null() => {
                    let mut name_buf = vec![0u16; 512];
                    let copied =
                        GetModuleBaseNameW(handle, None, &mut name_buf);
                    let _ = CloseHandle(handle);
                    if copied > 0 {
                        String::from_utf16_lossy(&name_buf[..copied as usize])
                    } else {
                        String::new()
                    }
                }
                _ => String::new(),
            }
        } else {
            String::new()
        };

        Ok(Some(WindowInfo {
            title,
            process_name,
            rect,
        }))
    }
}

#[cfg(not(target_os = "windows"))]
pub fn get_foreground_window() -> Result<Option<WindowInfo>, CaptureError> {
    // macOS: NSWorkspace.frontmostApplication (Phase 4).
    // Linux:  xprop -root _NET_ACTIVE_WINDOW (Phase 5).
    Ok(None)
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn info(title: &str, proc: &str) -> WindowInfo {
        WindowInfo {
            title: title.into(),
            process_name: proc.into(),
            rect: WindowRect { x: 0, y: 0, w: 1, h: 1 },
        }
    }

    #[test]
    fn phantom_is_wallet() {
        let v = classify(&info("Phantom — Receive", "phantom.exe"));
        assert_eq!(v, SensitivityVerdict::CryptoWallet);
    }

    #[test]
    fn metamask_in_title_is_wallet() {
        let v = classify(&info(
            "MetaMask Notification — Mozilla Firefox",
            "firefox.exe",
        ));
        assert_eq!(v, SensitivityVerdict::CryptoWallet);
    }

    #[test]
    fn vscode_with_env_is_secrets_editor() {
        let v = classify(&info(".env — myproject - Visual Studio Code", "code.exe"));
        assert_eq!(v, SensitivityVerdict::SecretsEditor);
    }

    #[test]
    fn vscode_without_env_is_not_sensitive() {
        let v = classify(&info("README.md — myproject - Visual Studio Code", "code.exe"));
        assert_eq!(v, SensitivityVerdict::NotSensitive);
    }

    #[test]
    fn notepad_with_secrets_is_sensitive() {
        let v = classify(&info("Untitled - Notepad - secrets.txt", "notepad.exe"));
        assert_eq!(v, SensitivityVerdict::SecretsEditor);
    }

    #[test]
    fn powershell_is_always_sensitive() {
        let v = classify(&info("Windows PowerShell", "powershell.exe"));
        assert_eq!(v, SensitivityVerdict::Terminal);
    }

    #[test]
    fn discord_is_not_sensitive() {
        let v = classify(&info("Discord — #general", "discord.exe"));
        assert_eq!(v, SensitivityVerdict::NotSensitive);
    }
}
