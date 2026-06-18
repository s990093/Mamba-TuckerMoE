// PerfWindow.swift — native window showing the live perf matrix (profiler
// dashboard: GPU / CPU / memory / tok-s). A distinct, opaque, titled window
// that pops up centered — clearly separate from the pet, not overlaid on it.
// Proves the model runs locally and compute-bound on Apple Silicon.

import Cocoa
import WebKit

final class PerfWindow: NSObject {
    let window: NSWindow
    private let webView: WKWebView
    private let url: URL

    init(url: URL) {
        self.url = url
        let dark = NSColor(calibratedWhite: 0.07, alpha: 1)
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 560, height: 440),
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered, defer: false)
        window.title = "Mamba · Perf Matrix"
        window.isReleasedWhenClosed = false
        window.level = .floating              // stay visible as a monitor
        window.appearance = NSAppearance(named: .darkAqua)
        window.isOpaque = true                // a solid, distinct window
        window.backgroundColor = dark

        webView = WKWebView(frame: window.contentLayoutRect)
        webView.autoresizingMask = [.width, .height]
        if #available(macOS 12.0, *) { webView.underPageBackgroundColor = dark }
        window.contentView = webView
        super.init()
    }

    // Single reused window: show centered if hidden, hide if visible.
    func toggle() {
        if window.isVisible {
            window.orderOut(nil)
        } else {
            webView.load(URLRequest(url: url))   // (re)connect to the live metrics WS
            window.center()
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        }
    }
}
