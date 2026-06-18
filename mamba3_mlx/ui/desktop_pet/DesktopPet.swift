// DesktopPet.swift — macOS transparent always-on-top wrapper for the Mamba mascot.
//
// Wraps the existing eyes page (served by the profiler FastAPI server) in a
// borderless, transparent, floating WKWebView so the SVG mascot sits directly
// on the desktop like a VTuber overlay — no window, no background.
//
// The target is split across files (compiled together as one module):
//   DesktopPet.swift   — AppDelegate, menu, gaze, zoom, message handlers, main
//   PetWindow.swift    — borderless transparent free-drag window
//   EmailWindow.swift  — native streaming email-draft window (markdown card)
//
// Build & run (see run.sh) — needs `make chat` running first (serves :7860).
// Info.plist is embedded so macOS prompts for microphone access:
//   swiftc *.swift -o pet -framework Cocoa -framework WebKit -framework AVFoundation \
//     -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker Info.plist
//   ./pet                       # loads http://127.0.0.1:7860/eyes?pet=1
//
// Controls:
//   drag anywhere    move the pet (a plain click still reaches the page)
//   menu bar 🐍 icon  Track cursor · Switch character · Click-through · Reload · Quit
//   the mascot's eyes follow the desktop cursor in real time
//
import Cocoa
import WebKit
import AVFoundation
import PDFKit

// MARK: - App

// Timestamped stdout logging so the `make pet` terminal shows what's happening.
func plog(_ s: String) {
    let t = ISO8601DateFormatter().string(from: Date())
    print("[pet \(t.suffix(13).prefix(8))] \(s)")
    fflush(stdout)
}

// Encode a Swift string as a safe JS string literal for evaluateJavaScript.
func jsLit(_ s: String) -> String {
    let data = (try? JSONSerialization.data(withJSONObject: s, options: [.fragmentsAllowed])) ?? Data("\"\"".utf8)
    return String(data: data, encoding: .utf8) ?? "\"\""
}

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler {
    var window: PetWindow!
    var webView: WKWebView!
    var statusItem: NSStatusItem!
    var gazeTimer: Timer?

    var trackingEnabled = true
    var clickThrough = false
    var currentChar = "eyes"
    var emailWindow: EmailWindow?
    var perfWindow: PerfWindow?          // single reused window, like EmailWindow

    let url: URL
    let size: NSSize

    init(url: URL, size: NSSize) {
        self.url = url
        self.size = size
    }

    func applicationDidFinishLaunching(_ note: Notification) {
        requestMicrophone() // trigger the TCC prompt up front, before the page asks

        let cfg = WKWebViewConfiguration()
        
        // Bridge the page's console.log/warn/error (and uncaught errors) to the
        // terminal so chat / WebSocket activity is visible for debugging.
        let ucc = WKUserContentController()
        ucc.add(self, name: "petlog")
        ucc.add(self, name: "petzoom")  // settings-panel "Pet size" buttons
        ucc.add(self, name: "petquit")  // settings-panel "Quit pet" button
        ucc.add(self, name: "petemail") // stream email draft to a native window
        ucc.add(self, name: "petfile")  // dropped file bytes → read & summarize
        let bridge = """
        (function () {
          function send(level, args) {
            try {
              window.webkit.messageHandlers.petlog.postMessage(
                level + ": " + Array.from(args).map(function (a) {
                  try { return typeof a === "object" ? JSON.stringify(a) : String(a); }
                  catch (e) { return String(a); }
                }).join(" "));
            } catch (e) {}
          }
          ["log", "warn", "error", "info"].forEach(function (k) {
            var orig = console[k] ? console[k].bind(console) : function () {};
            console[k] = function () { send(k, arguments); orig.apply(console, arguments); };
          });
          window.addEventListener("error", function (e) {
            var loc = (e.filename || "") + ":" + (e.lineno || 0) + ":" + (e.colno || 0);
            var stack = (e.error && e.error.stack) ? "\n" + e.error.stack : "";
            send("error", [(e.message || "uncaught error") + " @ " + loc + stack]);
          });
          window.addEventListener("unhandledrejection", function (e) {
            var r = e.reason;
            send("error", ["unhandled promise: " + ((r && r.stack) ? r.stack : r)]);
          });
        })();
        """
        ucc.addUserScript(WKUserScript(source: bridge, injectionTime: .atDocumentStart, forMainFrameOnly: false))
        cfg.userContentController = ucc

        plog("launch → \(url.absoluteString)  size \(Int(size.width))x\(Int(size.height))")
        webView = WKWebView(frame: NSRect(origin: .zero, size: size), configuration: cfg)
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.autoresizingMask = [.width, .height] // follow window on zoom
        webView.setValue(false, forKey: "drawsBackground") // private but stable
        if #available(macOS 12.0, *) { webView.underPageBackgroundColor = .clear }
        webView.load(URLRequest(url: url))

        window = PetWindow(
            contentRect: NSRect(origin: .zero, size: size),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.level = .floating
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        window.ignoresMouseEvents = false
        window.contentView = webView
        window.onDragChange = { [weak self] on in
            plog(on ? "drag start" : "drag end")
            self?.webView.evaluateJavaScript(
                "window.petDragging&&window.petDragging(\(on))", completionHandler: nil)
        }
        window.onShake = { [weak self] in
            plog("shake → dizzy")
            self?.webView.evaluateJavaScript(
                "window.petReact&&window.petReact('dizzy',1500)", completionHandler: nil)
        }

        positionBottomRight()
        window.makeKeyAndOrderFront(nil)

        setupStatusItem()
        startGazeTracking()
        NSApp.activate(ignoringOtherApps: true)
    }

    // Ask for microphone access at startup. Needs NSMicrophoneUsageDescription
    // in the embedded Info.plist (see run.sh -sectcreate) or macOS denies it
    // silently and the page's getUserMedia / mic level never starts.
    func requestMicrophone() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            plog("mic: already authorized")
        case .notDetermined:
            plog("mic: requesting permission…")
            AVCaptureDevice.requestAccess(for: .audio) { ok in
                plog("mic: permission \(ok ? "granted ✓" : "DENIED")")
            }
        case .denied, .restricted:
            plog("mic: DENIED — enable in System Settings ▸ Privacy ▸ Microphone")
        @unknown default:
            break
        }
    }

    func positionBottomRight() {
        guard let screen = NSScreen.main else { return }
        let vf = screen.visibleFrame
        let margin: CGFloat = 24
        window.setFrameOrigin(NSPoint(x: vf.maxX - size.width - margin,
                                      y: vf.minY + margin))
    }

    // MARK: Eye-to-cursor tracking
    //
    // Poll the global desktop cursor (~30fps) and feed a normalised direction to
    // the page. No accessibility permission needed — NSEvent.mouseLocation is a
    // plain class property.

    func startGazeTracking() {
        gazeTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            self?.updateGaze()
        }
    }

    func updateGaze() {
        guard trackingEnabled, webView != nil else { return }
        let mouse = NSEvent.mouseLocation                 // screen coords, y up
        let f = window.frame
        let eyeX = f.midX
        let eyeY = f.minY + f.height * 0.62               // eyes sit in upper area
        let radius = 190.0                                // smaller → eyes deflect sooner / more
        var nx = Double(mouse.x - eyeX) / radius
        var ny = Double(mouse.y - eyeY) / radius
        nx = max(-1, min(1, nx))
        ny = max(-1, min(1, ny))
        let cssY = -ny                                    // CSS y is top-down
        webView.evaluateJavaScript(
            "window.petLookAt&&window.petLookAt(\(nx),\(cssY))", completionHandler: nil)
    }

    // MARK: Menu bar (no Dock icon — this is settings + how you quit)

    // Holds the verified demo-prompt submenu (rebuilt after /api/demo-config fetch).
    var demoMenuItem: NSMenuItem?

    func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.title = "🐍"
        let menu = NSMenu()

        let track = item("Track cursor", #selector(toggleTracking)); track.state = .on
        menu.addItem(track)
        menu.addItem(item("Switch character", #selector(switchCharacter), key: "x"))
        menu.addItem(item("Switch persona (system prompt)", #selector(switchPersona), key: "c"))
        menu.addItem(.separator())

        // Demo prompts submenu (verified self_awareness + email_summary).
        // Populated asynchronously from /api/demo-config after the menu is built.
        let demoTop = NSMenuItem(title: "Demo prompts", action: nil, keyEquivalent: "")
        let placeholder = NSMenu()
        placeholder.addItem(NSMenuItem(title: "Loading…", action: nil, keyEquivalent: ""))
        demoTop.submenu = placeholder
        menu.addItem(demoTop)
        demoMenuItem = demoTop
        menu.addItem(.separator())

        menu.addItem(item("Bigger", #selector(bigger), key: "="))
        menu.addItem(item("Smaller", #selector(smaller), key: "-"))
        let ct = item("Click-through", #selector(toggleClickThrough)); ct.state = .off
        menu.addItem(ct)
        menu.addItem(item("Reset settings", #selector(resetSettings)))
        menu.addItem(item("Perf matrix", #selector(togglePerf), key: "p"))
        menu.addItem(.separator())
        menu.addItem(item("Refresh demo prompts", #selector(refreshDemoPrompts)))
        menu.addItem(item("Test TTS (say 'Mamba ready')", #selector(testTTS)))
        menu.addItem(item("Reload", #selector(reload), key: "r"))
        menu.addItem(NSMenuItem(title: "Quit Pet",
                                action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        statusItem.menu = menu

        // Fire fetch — non-blocking; populates demoMenuItem when /api/demo-config
        // (served by `make chat`) responds.
        fetchAndBuildDemoMenu()
    }

    // MARK: Demo-prompt menu (fetched live from chat_demo /api/demo-config)

    @objc func refreshDemoPrompts() {
        plog("refresh demo prompts")
        fetchAndBuildDemoMenu()
    }

    /// Diagnostic helper: speak a fixed phrase via the page's SpeechSynthesis.
    /// Kept as a menu item because it's a quick check that the WebKit audio
    /// path is alive (separate from the chat WS pipeline).
    @objc func testTTS() {
        plog("TTS test → say 'Mamba ready'")
        let js = """
        (function(){
          try {
            var u = new SpeechSynthesisUtterance('Mamba ready');
            u.rate = 1.0; u.pitch = 1.0;
            speechSynthesis.cancel();
            speechSynthesis.speak(u);
            return true;
          } catch(e) { return false; }
        })()
        """
        webView.evaluateJavaScript(js, completionHandler: { res, err in
            if let err = err { plog("TTS test JS error: \(err.localizedDescription)") }
            else if let b = res as? Bool { plog("TTS test → \(b ? "queued" : "failed")") }
        })
    }

    /// Fetch /api/demo-config from the running chat_demo server, then rebuild
    /// the "Demo prompts" submenu so each item carries its (category, seed,
    /// prompt) triple and can fire via `window.petSendDemo`.
    private func fetchAndBuildDemoMenu() {
        // The same host the WKWebView already loaded — strip path, hit /api/demo-config.
        guard var comps = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return }
        comps.path = "/api/demo-config"
        comps.query = nil
        guard let endpoint = comps.url else { return }
        plog("fetch demo-config → \(endpoint.absoluteString)")
        let task = URLSession.shared.dataTask(with: endpoint) { [weak self] data, _, err in
            guard let self = self else { return }
            if let err = err {
                plog("demo-config fetch failed: \(err.localizedDescription)")
                return
            }
            guard let data = data,
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let cats = obj["categories"] as? [[String: Any]]
            else {
                plog("demo-config: malformed JSON")
                return
            }
            DispatchQueue.main.async { self.populateDemoMenu(from: cats) }
        }
        task.resume()
    }

    private func populateDemoMenu(from categories: [[String: Any]]) {
        guard let top = demoMenuItem else { return }
        let sub = NSMenu()
        var total = 0
        for cat in categories {
            guard let catKey = cat["key"] as? String,
                  let examples = cat["examples"] as? [[String: Any]],
                  !examples.isEmpty
            else { continue }
            let catTitle = (cat["title"] as? String) ?? catKey
            let header = NSMenuItem(title: "\(catTitle) (\(examples.count))",
                                    action: nil, keyEquivalent: "")
            header.isEnabled = false
            sub.addItem(header)
            for ex in examples {
                guard let user = ex["user"] as? String, !user.isEmpty else { continue }
                let sampling = (ex["sampling"] as? [String: Any]) ?? [:]
                let seed = (sampling["seed"] as? Int) ?? -1
                let label = user.count > 70 ? String(user.prefix(70)) + "…" : user
                let it = NSMenuItem(title: "  \(label)",
                                    action: #selector(demoPromptClicked(_:)),
                                    keyEquivalent: "")
                it.target = self
                // Pack the triple via a dictionary on representedObject.
                it.representedObject = [
                    "catKey": catKey,
                    "seed":   seed,
                    "prompt": user,
                ] as NSDictionary
                it.toolTip = "seed=\(seed) · \(catKey)"
                sub.addItem(it)
                total += 1
            }
            sub.addItem(.separator())
        }
        if total == 0 {
            sub.addItem(NSMenuItem(title: "(no demo prompts found)", action: nil, keyEquivalent: ""))
        }
        top.submenu = sub
        plog("demo-config loaded → \(total) prompts across \(categories.count) categories")
    }

    @objc func demoPromptClicked(_ sender: NSMenuItem) {
        guard let info = sender.representedObject as? NSDictionary,
              let catKey = info["catKey"] as? String,
              let prompt = info["prompt"] as? String
        else { return }
        let seed = info["seed"] as? Int ?? -1
        // JS-escape via JSON encoding — survives quotes, newlines, unicode.
        func jsString(_ s: String) -> String {
            let data = (try? JSONSerialization.data(withJSONObject: [s], options: []))
                ?? Data("[\"\"]".utf8)
            let raw  = String(data: data, encoding: .utf8) ?? "[\"\"]"
            // raw is "[\"...\"]"; strip outer brackets.
            return String(raw.dropFirst().dropLast())
        }
        let js = "window.petSendDemo && window.petSendDemo(\(jsString(catKey)),\(seed),\(jsString(prompt)))"
        plog("demo-send cat=\(catKey) seed=\(seed) prompt=\"\(prompt.prefix(60))…\"")
        webView.evaluateJavaScript(js, completionHandler: { _, err in
            if let err = err {
                plog("petSendDemo JS error: \(err.localizedDescription)")
            }
        })
    }

    private func item(_ title: String, _ sel: Selector, key: String = "") -> NSMenuItem {
        let it = NSMenuItem(title: title, action: sel, keyEquivalent: key)
        it.target = self
        return it
    }

    @objc func toggleTracking(_ sender: NSMenuItem) {
        trackingEnabled.toggle()
        sender.state = trackingEnabled ? .on : .off
        plog("track cursor → \(trackingEnabled ? "on" : "off")")
        if !trackingEnabled {
            webView.evaluateJavaScript("window.petLookAt&&window.petLookAt(0,0)")
        }
    }

    @objc func toggleClickThrough(_ sender: NSMenuItem) {
        clickThrough.toggle()
        window.ignoresMouseEvents = clickThrough
        sender.state = clickThrough ? .on : .off
        plog("click-through → \(clickThrough ? "on" : "off")")
    }

    // Drive the page's own (hidden) character switch + reset buttons by clicking
    // them — the page functions are module-scoped, but the DOM controls aren't.
    @objc func switchCharacter() {
        currentChar = (currentChar == "eyes") ? "tars" : "eyes"
        plog("switch character → \(currentChar)")
        webView.evaluateJavaScript(
            "var b=document.querySelector('[data-char=\"\(currentChar)\"]');b&&b.click();")
    }

    // Cycle the model's system prompt / category — the page binds this to 'c'.
    @objc func switchPersona() {
        plog("switch persona (key c)")
        webView.evaluateJavaScript(
            "document.dispatchEvent(new KeyboardEvent('keydown',{key:'c',bubbles:true}));")
    }

    @objc func resetSettings() {
        plog("reset settings")
        webView.evaluateJavaScript("var b=document.getElementById('set-reset');b&&b.click();")
    }

    @objc func bigger() { zoom(1.15) }
    @objc func smaller() { zoom(1.0 / 1.15) }

    private var pageZoom: CGFloat = 1.0

    // Enlarge the *whole pet*: grow the window AND scale the page content (the
    // character is fixed-size, so resizing the window alone left it the same).
    private func zoom(_ k: CGFloat) {
        pageZoom = min(2.6, max(0.6, pageZoom * k))
        var f = window.frame
        let cx = f.midX, cy = f.midY
        let w = min(900, max(220, f.width * k))
        let h = min(1100, max(260, f.height * k))
        f.size = NSSize(width: w, height: h)
        f.origin = NSPoint(x: cx - w / 2, y: cy - h / 2)
        window.setFrame(f, display: true, animate: true)
        webView.evaluateJavaScript("document.documentElement.style.zoom='\(pageZoom)';")
        plog("zoom → window \(Int(w))x\(Int(h)) · content @\(String(format: "%.2f", pageZoom))×")
    }

    @objc func reload() { plog("reload"); webView.reload() }

    // A file was dropped on the page → it sends {name, ext, b64}. The base64
    // decode + PDFKit text extraction are HEAVY for large files and this handler
    // runs on the main thread, so do that work on a background queue (otherwise
    // the whole pet UI freezes). Show "reading" immediately for feedback.
    func handlePetFile(_ d: [String: Any]) {
        let name = (d["name"] as? String) ?? "document"
        let ext = ((d["ext"] as? String) ?? "").lowercased()
        guard let b64 = d["b64"] as? String else { plog("dropped file: bad payload"); return }

        plog("file dropped: \(name) — extracting (\(b64.count) b64 chars) …")
        webView.evaluateJavaScript("window.petReadingOn&&window.petReadingOn()") // instant feedback

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            var text: String?
            if let data = Data(base64Encoded: b64) {
                if ext == "pdf" {
                    text = PDFDocument(data: data)?.string     // PDFKit (slow on big PDFs)
                } else {
                    text = String(data: data, encoding: .utf8)
                }
            }
            let body = text?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            // Clip BEFORE handing to the model. 3500 chars ≈ ≤1500 tokens even for
            // dense code (~2.3 ch/tok), staying well under the 2048 ctx limit with
            // room for generation; also bounds prefill memory/latency.
            let clipped = body.count > 3500 ? String(body.prefix(3500)) : body

            DispatchQueue.main.async {
                if clipped.isEmpty {
                    plog("could not extract text from: \(name) (.\(ext))")
                    let reason = (ext == "pdf")
                        ? "no text in that PDF (scanned image?)"
                        : "couldn't read that file as text"
                    self.webView.evaluateJavaScript(
                        "window.petFileError&&window.petFileError(\(jsLit(reason)))")
                    return
                }
                plog("file → summarize: \(name) (\(body.count) chars → \(clipped.count))")
                self.webView.evaluateJavaScript(
                    "window.petSummarize&&window.petSummarize(\(jsLit(name)),\(jsLit(clipped)))")
            }
        }
    }

    // Native perf-matrix window (profiler dashboard :8765) — single reused
    // window, handled like the email window.
    @objc func togglePerf() {
        if perfWindow == nil {
            let host = url.host ?? "127.0.0.1"
            perfWindow = PerfWindow(url: URL(string: "http://\(host):8765/") ?? url)
        }
        plog("toggle perf matrix")
        perfWindow?.toggle()
    }

    // MARK: Page console / errors → terminal

    func userContentController(_ uc: WKUserContentController, didReceive message: WKScriptMessage) {
        switch message.name {
        case "petlog":
            let s = "\(message.body)"
            // Make page errors/warnings stand out from the token-log stream.
            if s.hasPrefix("error") || s.hasPrefix("warn") {
                plog("‼️  [page] \(s)")
            } else {
                plog("[page] \(s)")
            }
        case "petzoom":
            let dir = (message.body as? String) ?? ""
            plog("pet size button → \(dir)")
            zoom(dir == "in" ? 1.15 : 1.0 / 1.15)
        case "petquit":
            plog("quit (settings button)")
            NSApp.terminate(nil)
        case "petfile":
            if let d = message.body as? [String: Any] { handlePetFile(d) }
        case "petemail":
            guard let d = message.body as? [String: Any],
                  let type = d["type"] as? String else { break }
            if emailWindow == nil { emailWindow = EmailWindow() }
            switch type {
            case "start": plog("email window: start"); emailWindow?.begin()
            case "token": emailWindow?.append(d["text"] as? String ?? "")
            case "done":  plog("email window: done");  emailWindow?.finish(d["full"] as? String ?? "")
            default: break
            }
        default:
            break
        }
    }

    // Grant in-page mic capture so voice features can work (TTS always works;
    // note that webkitSpeechRecognition itself is unreliable inside WKWebView).
    @available(macOS 12.0, *)
    func webView(_ webView: WKWebView,
                 requestMediaCapturePermissionFor origin: WKSecurityOrigin,
                 initiatedByFrame frame: WKFrameInfo,
                 type: WKMediaCaptureType,
                 decisionHandler: @escaping (WKPermissionDecision) -> Void) {
        decisionHandler(.grant)
    }

    // The email "Send" button does window.open(gmail compose). WKWebView drops
    // new windows by default, so open the URL in the user's real browser.
    func webView(_ webView: WKWebView,
                 createWebViewWith configuration: WKWebViewConfiguration,
                 for navigationAction: WKNavigationAction,
                 windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = navigationAction.request.url {
            plog("open in browser → \(url.absoluteString)")
            NSWorkspace.shared.open(url)
        }
        return nil
    }

    // Keep the pet on its own page: any external (non-localhost) link the page
    // tries to navigate to opens in the default browser instead.
    func webView(_ webView: WKWebView,
                 decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if navigationAction.navigationType == .linkActivated,
           let url = navigationAction.request.url,
           let host = url.host, host != "127.0.0.1", host != "localhost" {
            plog("external link → browser: \(url.absoluteString)")
            NSWorkspace.shared.open(url)
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        plog("page loaded ✓ \(webView.url?.absoluteString ?? "")")
        webView.evaluateJavaScript(
            "document.documentElement.style.background='transparent';" +
            "document.body.style.background='transparent';" +
            "document.documentElement.style.zoom='\(pageZoom)';")
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        plog("page load FAILED: \(error.localizedDescription)")
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        plog("page load FAILED (provisional): \(error.localizedDescription)")
    }
}
