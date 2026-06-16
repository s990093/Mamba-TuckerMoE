// main.swift — entry point + CLI args. Top-level executable code must
// live in a file literally named main.swift when the target is multi-file.

import Cocoa

// MARK: - Args

func parseArgs() -> (URL, NSSize) {
    var urlStr = "http://127.0.0.1:7860/eyes?pet=1" // chat_demo (make chat) — real model /ws
    var w: CGFloat = 360, h: CGFloat = 440
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let a = it.next() {
        switch a {
        case "--url": if let v = it.next() { urlStr = v }
        case "--width": if let v = it.next(), let n = Double(v) { w = CGFloat(n) }
        case "--height": if let v = it.next(), let n = Double(v) { h = CGFloat(n) }
        default: break
        }
    }
    guard let url = URL(string: urlStr) else {
        FileHandle.standardError.write(Data("Invalid --url\n".utf8)); exit(1)
    }
    return (url, NSSize(width: w, height: h))
}

let (petURL, petSize) = parseArgs()
let app = NSApplication.shared
app.setActivationPolicy(.accessory) // no Dock icon, no app menu
let delegate = AppDelegate(url: petURL, size: petSize)
app.delegate = delegate
app.run()
