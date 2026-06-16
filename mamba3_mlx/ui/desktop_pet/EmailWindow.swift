// EmailWindow.swift — native macOS window that streams an email draft
// token-by-token, styled to match the eyes UI. Part of the desktop-pet target.

import Cocoa
import WebKit

// MARK: - Native email window
//
// A separate, normal macOS window that shows an email draft streaming in
// token-by-token (typewriter), with Copy / Open-in-Mail. Used instead of the
// in-page card so the draft doesn't appear all at once crammed on the pet.

final class EmailWindow: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
    let window: NSWindow
    private let webView: WKWebView
    private var fullText = ""
    private var ready = false

    override init() {
        let w: CGFloat = 460, h: CGFloat = 560
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: w, height: h),
            styleMask: [.titled, .closable, .resizable, .miniaturizable, .fullSizeContentView],
            backing: .buffered, defer: false)
            
        window.title = "Email Draft"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.isReleasedWhenClosed = false
        window.isMovableByWindowBackground = true
        window.level = .floating
        window.appearance = NSAppearance(named: .darkAqua)

        // Dark blur behind the (transparent) web card → matches the original
        // email card's translucent backdrop-filter look.
        let vfx = NSVisualEffectView(frame: NSRect(x: 0, y: 0, width: w, height: h))
        vfx.material = .hudWindow
        vfx.blendingMode = .behindWindow
        vfx.state = .active
        vfx.autoresizingMask = [.width, .height]

        let cfg = WKWebViewConfiguration()
        let ucc = WKUserContentController()
        cfg.userContentController = ucc
        webView = WKWebView(frame: vfx.bounds, configuration: cfg)
        webView.autoresizingMask = [.width, .height]
        webView.setValue(false, forKey: "drawsBackground")
        vfx.addSubview(webView)
        window.contentView = vfx
        super.init()

        ucc.add(self, name: "emailact")
        webView.navigationDelegate = self
        webView.loadHTMLString(Self.html, baseURL: nil)
    }

    func begin() {
        fullText = ""
        webView.evaluateJavaScript("window.reset&&window.reset()")
        if !window.isVisible { window.center() }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func append(_ s: String) {
        fullText += s
        if ready {
            webView.evaluateJavaScript("window.addToken&&window.addToken(\(Self.jsLit(s)))")
        }
    }

    func finish(_ full: String) {
        if !full.isEmpty { fullText = full }
        webView.evaluateJavaScript("window.setFull&&window.setFull(\(Self.jsLit(fullText)))")
    }

    // Flush whatever streamed in before the page finished loading.
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        ready = true
        if !fullText.isEmpty {
            webView.evaluateJavaScript("window.setFull&&window.setFull(\(Self.jsLit(fullText)))")
        }
    }

    // Copy / Open-in-Mail come from the styled in-card buttons.
    func userContentController(_ uc: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.name == "emailact",
              let d = message.body as? [String: Any],
              let action = d["a"] as? String else { return }
        if let f = d["full"] as? String { fullText = f }
        let (subj, body) = parsed()
        switch action {
        case "copy":
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString("Subject: \(subj)\n\n\(body)", forType: .string)
        case "send":
            let su = subj.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
            let bo = body.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
            if let url = URL(string: "https://mail.google.com/mail/u/0/?tf=cm&to=&su=\(su)&body=\(bo)") {
                NSWorkspace.shared.open(url)
            }
        default: break
        }
    }

    private func parsed() -> (subject: String, body: String) {
        let t = fullText.trimmingCharacters(in: .whitespacesAndNewlines)
        var lines = t.components(separatedBy: "\n")
        var subject = ""
        if let first = lines.first {
            if let r = first.range(of: "subject:", options: .caseInsensitive) {
                subject = String(first[r.upperBound...]).trimmingCharacters(in: .whitespaces)
                lines.removeFirst()
            } else {
                subject = first
            }
        }
        let body = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        // Export clean plain text (the card shows rendered markdown, but a real
        // email shouldn't contain ** __ ` # etc.).
        return (stripMd(subject), stripMd(body))
    }

    private func stripMd(_ s: String) -> String {
        func re(_ str: String, _ pat: String, _ rep: String) -> String {
            str.replacingOccurrences(of: pat, with: rep, options: .regularExpression)
        }
        var t = s
        t = re(t, #"\[([^\]]+)\]\([^)]+\)"#, "$1")   // [text](url) → text
        t = t.replacingOccurrences(of: "**", with: "")
        t = t.replacingOccurrences(of: "__", with: "")
        t = t.replacingOccurrences(of: "`", with: "")
        t = re(t, "(?m)^\\s{0,3}#{1,6}\\s+", "")        // headings
        t = re(t, "(?m)^\\s*[-*]\\s+", "• ")            // bullets
        t = re(t, "(?m)^\\s*---\\s*$", "")              // horizontal rules
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // Encode a Swift string as a safe JS string literal for evaluateJavaScript.
    private static func jsLit(_ s: String) -> String {
        let data = (try? JSONSerialization.data(withJSONObject: s, options: [.fragmentsAllowed])) ?? Data("\"\"".utf8)
        return String(data: data, encoding: .utf8) ?? "\"\""
    }

    // Self-contained email card matching the eyes UI (warm accent, dark glass).
    // Body is rendered as Markdown; the Edit toggle reveals the raw source.
    private static let html = #"""
    <!doctype html><html><head><meta charset="utf-8"><style>
      :root{ --accent:#cc785c; --text:rgba(238,236,226,.9); }
      *{ box-sizing:border-box; margin:0; }
      html,body{ height:100%; background:transparent;
        font-family:-apple-system,"Inter","Segoe UI",sans-serif; color:var(--text);
        -webkit-font-smoothing:antialiased; user-select:text; }
      .card{ display:flex; flex-direction:column; height:100%;
        background:rgba(20,18,15,.82); }
      .hdr{ display:flex; align-items:center; justify-content:flex-end; gap:8px;
        padding:10px 14px 10px 78px;
        border-bottom:1px solid rgba(255,255,255,.07);
        background:linear-gradient(180deg,rgba(204,120,92,.10),transparent); }
      .ttl{ margin-right:auto; font-size:12px; letter-spacing:.08em;
        text-transform:uppercase; color:rgba(238,236,226,.5); }
      .btn{ font-size:12px; padding:6px 12px; border-radius:8px;
        border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.05);
        color:rgba(238,236,226,.85); cursor:pointer; transition:background .15s; }
      .btn:hover{ background:rgba(255,255,255,.10); }
      .btn.send{ background:var(--accent); border-color:var(--accent);
        color:#1a120e; font-weight:600; }
      .btn.send:hover{ background:#d9876b; }
      .subjrow{ padding:16px 20px 4px; }
      .lbl{ font-size:11px; letter-spacing:.1em; text-transform:uppercase;
        color:var(--accent); opacity:.85; }
      .subj{ margin-top:5px; font-size:17px; font-weight:600;
        color:rgba(245,240,232,.96); min-height:22px; }
      .sep{ height:1px; background:rgba(255,255,255,.07); margin:12px 20px; }
      .body{ flex:1; overflow:auto; padding:2px 20px 22px; font-size:14px;
        line-height:1.65; word-break:break-word; }
      .body div{ min-height:1.1em; }
      .body strong{ font-weight:700; color:rgba(245,240,232,.98); }
      .body em{ font-style:italic; }
      .body code{ font-family:ui-monospace,Menlo,monospace; font-size:12.5px;
        background:rgba(255,255,255,.08); padding:1px 5px; border-radius:4px; }
      .body h3{ font-size:15px; margin:10px 0 4px; color:rgba(245,240,232,.96); }
      .body hr{ border:none; border-top:1px solid rgba(255,255,255,.12); margin:12px 0; }
      .body ul{ margin:4px 0 4px 4px; padding-left:18px; }
      .body li{ margin:2px 0; }
      .body a{ color:var(--accent); }
      #ta{ width:100%; height:100%; min-height:320px; resize:none; border:none;
        outline:none; background:rgba(0,0,0,.18); color:var(--text); font:inherit;
        line-height:1.6; padding:10px 12px; border-radius:8px; white-space:pre-wrap; }
      .cur{ display:inline-block; width:7px; height:15px; background:var(--accent);
        vertical-align:text-bottom; margin-left:1px; animation:blink 1s steps(1) infinite; }
      @keyframes blink{ 50%{ opacity:0; } }
      ::-webkit-scrollbar{ width:8px; }
      ::-webkit-scrollbar-thumb{ background:rgba(255,255,255,.14); border-radius:4px; }
    </style></head><body>
      <div class="card">
        <div class="hdr">
          <span class="ttl">✦ Email Draft</span>
          <button class="btn" id="edit">Edit</button>
          <button class="btn" id="copy">Copy</button>
          <button class="btn send" id="send">Open in Mail</button>
        </div>
        <div class="subjrow"><div class="lbl">Subject</div><div class="subj" id="subj"></div></div>
        <div class="sep"></div>
        <div class="body" id="body"></div>
      </div>
      <script>
        var raw="", done=false, editing=false;
        function esc(s){ return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
        function inl(s){
          return s
            .replace(/`([^`]+)`/g,"<code>$1</code>")
            .replace(/\*\*([^*]+?)\*\*/g,"<strong>$1</strong>")
            .replace(/__([^_]+?)__/g,"<strong>$1</strong>")
            .replace(/(^|[^*])\*([^*\n]+?)\*/g,"$1<em>$2</em>")
            .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g,'<a href="$2">$1</a>');
        }
        function md(src){
          var lines=esc(src).split("\n"), out=[], list=false;
          for(var i=0;i<lines.length;i++){
            var ln=lines[i];
            if(/^\s*---\s*$/.test(ln)){ if(list){out.push("</ul>");list=false;} out.push("<hr>"); continue; }
            var hm=ln.match(/^\s*(#{1,3})\s+(.*)$/);
            if(hm){ if(list){out.push("</ul>");list=false;} out.push("<h3>"+inl(hm[2])+"</h3>"); continue; }
            var lm=ln.match(/^\s*[-*]\s+(.*)$/);
            if(lm){ if(!list){out.push("<ul>");list=true;} out.push("<li>"+inl(lm[1])+"</li>"); continue; }
            if(list){ out.push("</ul>"); list=false; }
            if(ln.trim()===""){ out.push("<br>"); } else { out.push("<div>"+inl(ln)+"</div>"); }
          }
          if(list) out.push("</ul>");
          return out.join("");
        }
        function split(){
          var t=raw.replace(/^\s+/,"");
          var m=t.match(/^subject:\s*(.*?)(?:\n([\s\S]*))?$/i);
          return m ? { subj:(m[1]||"").trim(), body:(m[2]||"") } : { subj:"", body:t };
        }
        function render(){
          if(editing) return;
          var p=split();
          document.getElementById("subj").innerHTML=inl(esc(p.subj));
          var b=document.getElementById("body");
          b.innerHTML=md(p.body);
          if(!done){ var c=document.createElement("span"); c.className="cur"; b.appendChild(c); }
          b.scrollTop=b.scrollHeight;
        }
        window.reset=function(){ raw=""; done=false; editing=false;
          document.getElementById("edit").textContent="Edit"; render(); };
        window.addToken=function(s){ raw+=s; render(); };
        window.setFull=function(s){ raw=s; done=true; render(); };
        function toggleEdit(){
          var b=document.getElementById("body"), eb=document.getElementById("edit");
          if(!editing){
            editing=true; eb.textContent="Done";
            b.innerHTML='<textarea id="ta" spellcheck="false"></textarea>';
            var ta=document.getElementById("ta"); ta.value=raw; ta.focus();
          } else {
            var ta=document.getElementById("ta"); if(ta) raw=ta.value;
            editing=false; done=true; eb.textContent="Edit"; render();
          }
        }
        document.getElementById("edit").onclick=toggleEdit;
        document.getElementById("copy").onclick=function(){
          try{ window.webkit.messageHandlers.emailact.postMessage({a:"copy",full:raw}); }catch(e){} };
        document.getElementById("send").onclick=function(){
          try{ window.webkit.messageHandlers.emailact.postMessage({a:"send",full:raw}); }catch(e){} };
        render();
      </script>
    </body></html>
    """#
}
