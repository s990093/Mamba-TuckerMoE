// PetWindow.swift — borderless, transparent, free-drag window for the pet.
// Part of the desktop-pet target (compiled together via run.sh).

import Cocoa

// MARK: - Borderless window with threshold-based free dragging
//
// A WKWebView covers the whole window, so we can't rely on
// isMovableByWindowBackground. Instead we watch the mouse at the window level:
// a press that moves past a small threshold becomes a window drag; a press that
// is released in place stays a normal click and reaches the web page.

final class PetWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }

    var onDragChange: ((Bool) -> Void)?
    var onShake: (() -> Void)?

    private var downMouse: NSPoint?
    private var downOrigin: NSPoint?
    private var dragging = false

    // Shake detection: rapid left/right reversals while dragging → "dizzy".
    private var lastDragX: CGFloat = 0
    private var lastMoveSign = 0
    private var shakeReversals = 0
    private var shakeWindowStart: TimeInterval = 0

    override func sendEvent(_ event: NSEvent) {
        switch event.type {
        case .leftMouseDown:
            downMouse = NSEvent.mouseLocation
            downOrigin = frame.origin
            dragging = false
            lastDragX = NSEvent.mouseLocation.x
            lastMoveSign = 0; shakeReversals = 0
            // Take focus so the page can monitor keys (e.g. hold-Space to talk)
            // only once you've clicked into the pet window.
            if !isKeyWindow {
                NSApp.activate(ignoringOtherApps: true)
                makeKeyAndOrderFront(nil)
            }
            super.sendEvent(event)
        case .leftMouseDragged:
            guard let dm = downMouse, let orig = downOrigin else {
                super.sendEvent(event); return
            }
            let cur = NSEvent.mouseLocation
            let dx = cur.x - dm.x, dy = cur.y - dm.y
            if !dragging && (dx * dx + dy * dy) > 9 {     // ~3px slop
                dragging = true
                onDragChange?(true)                       // "picked up" feedback
            }
            if dragging {
                setFrameOrigin(NSPoint(x: orig.x + dx, y: orig.y + dy))
                detectShake(cur.x)
                return // consume: don't forward drags into the page
            }
            super.sendEvent(event)
        case .leftMouseUp:
            let wasDragging = dragging
            downMouse = nil; downOrigin = nil; dragging = false
            if wasDragging { onDragChange?(false); return } // swallow the up
            super.sendEvent(event)
        default:
            super.sendEvent(event)
        }
    }

    // Count rapid horizontal direction reversals; enough within a short window
    // means the user is shaking the pet around.
    private func detectShake(_ x: CGFloat) {
        let moveDX = x - lastDragX
        guard abs(moveDX) > 5 else { return }
        let sign = moveDX > 0 ? 1 : -1
        if lastMoveSign != 0 && sign != lastMoveSign {
            let now = ProcessInfo.processInfo.systemUptime
            if now - shakeWindowStart > 0.7 { shakeReversals = 0; shakeWindowStart = now }
            shakeReversals += 1
            if shakeReversals >= 4 { shakeReversals = 0; onShake?() }
        }
        lastMoveSign = sign
        lastDragX = x
    }
}
