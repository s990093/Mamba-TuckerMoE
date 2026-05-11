import Foundation
import Metal

struct Args {
    var r3 = 256
    var r2 = 1024
    var e = 8
    var k = 2
    var warmup = 10
    var trials = 100
}

func parseArgs() -> Args {
    var a = Args()
    var i = 1
    let argv = CommandLine.arguments
    while i < argv.count {
        let key = argv[i]
        let val = (i + 1 < argv.count) ? argv[i + 1] : ""
        switch key {
        case "--r3": a.r3 = Int(val) ?? a.r3; i += 2
        case "--r2": a.r2 = Int(val) ?? a.r2; i += 2
        case "--e": a.e = Int(val) ?? a.e; i += 2
        case "--k": a.k = Int(val) ?? a.k; i += 2
        case "--warmup": a.warmup = Int(val) ?? a.warmup; i += 2
        case "--trials": a.trials = Int(val) ?? a.trials; i += 2
        default: i += 1
        }
    }
    return a
}

func fillRandom(_ ptr: UnsafeMutablePointer<Float>, count: Int) {
    for i in 0..<count {
        ptr[i] = Float.random(in: -1.0...1.0)
    }
}

func reducePartial(_ partial: UnsafePointer<Float>, out: UnsafeMutablePointer<Float>, k: Int, r2: Int) {
    for c in 0..<r2 {
        var s: Float = 0
        for kk in 0..<k {
            s += partial[kk * r2 + c]
        }
        out[c] = s
    }
}

func main() throws {
    let args = parseArgs()
    guard args.r3 % 32 == 0, args.r2 % 32 == 0 else {
        throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "r3/r2 must be multiples of 32"])
    }
    guard args.k > 0 && args.k <= args.e else {
        throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "k must be in [1, e]"])
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "No Metal device"])
    }
    guard let queue = device.makeCommandQueue() else {
        throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "No command queue"])
    }

    let metalPath = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("fused_latent_moe_async.metal")
        .path
    let src = try String(contentsOfFile: metalPath, encoding: .utf8)

    let options = MTLCompileOptions()
    if #available(macOS 15.0, *) {
        options.languageVersion = .version3_1
    }
    let library = try device.makeLibrary(source: src, options: options)

    func makePSO(_ name: String) throws -> MTLComputePipelineState {
        let fcv = MTLFunctionConstantValues()
        var r3 = UInt32(args.r3), r2 = UInt32(args.r2), e = UInt32(args.e), k = UInt32(args.k)
        fcv.setConstantValue(&r3, type: .uint, index: 0)
        fcv.setConstantValue(&r2, type: .uint, index: 1)
        fcv.setConstantValue(&e, type: .uint, index: 2)
        fcv.setConstantValue(&k, type: .uint, index: 3)
        let fn = try library.makeFunction(name: name, constantValues: fcv)
        return try device.makeComputePipelineState(function: fn)
    }

    let psoSync = try makePSO("fused_latent_moe_sync_partial")
    let psoAsync = try makePSO("fused_latent_moe_async_partial")

    let xCount = args.r3
    let gCount = args.e * args.r3 * args.r2
    let idxCount = args.k
    let partialCount = args.k * args.r2
    let outCount = args.r2

    let xBuf = device.makeBuffer(length: xCount * MemoryLayout<Float>.stride)!
    let gBuf = device.makeBuffer(length: gCount * MemoryLayout<Float>.stride)!
    let idxBuf = device.makeBuffer(length: idxCount * MemoryLayout<UInt32>.stride)!
    let partialBuf = device.makeBuffer(length: partialCount * MemoryLayout<Float>.stride)!
    let outHost = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
    defer { outHost.deallocate() }

    fillRandom(xBuf.contents().assumingMemoryBound(to: Float.self), count: xCount)
    fillRandom(gBuf.contents().assumingMemoryBound(to: Float.self), count: gCount)
    var indices = Array(0..<args.e).shuffled().prefix(args.k).map { UInt32($0) }
    idxBuf.contents().copyMemory(from: &indices, byteCount: idxCount * MemoryLayout<UInt32>.stride)

    let tg = MTLSize(width: 32, height: 1, depth: 1)
    let grid = MTLSize(width: args.r2 / 32, height: 1, depth: args.k)

    func run(_ pso: MTLComputePipelineState) throws {
        guard let cb = queue.makeCommandBuffer(),
              let ce = cb.makeComputeCommandEncoder() else {
            throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot make command encoder"])
        }
        ce.setComputePipelineState(pso)
        ce.setBuffer(xBuf, offset: 0, index: 0)
        ce.setBuffer(gBuf, offset: 0, index: 1)
        ce.setBuffer(idxBuf, offset: 0, index: 2)
        ce.setBuffer(partialBuf, offset: 0, index: 3)
        ce.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        ce.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        reducePartial(
            partialBuf.contents().assumingMemoryBound(to: Float.self),
            out: outHost,
            k: args.k,
            r2: args.r2
        )
    }

    for _ in 0..<args.warmup {
        try run(psoSync)
        try run(psoAsync)
    }

    func bench(_ pso: MTLComputePipelineState, trials: Int) throws -> Double {
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<trials { try run(pso) }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        return dt * 1000.0 / Double(max(trials, 1))
    }

    let msSync = try bench(psoSync, trials: args.trials)
    let msAsync = try bench(psoAsync, trials: args.trials)

    print("shape: R3=\(args.r3) R2=\(args.r2) E=\(args.e) K=\(args.k)")
    print(String(format: "pure_metal sync_partial+reduce : %.4f ms", msSync))
    print(String(format: "pure_metal async_partial+reduce: %.4f ms", msAsync))
    print(String(format: "speedup async/sync: %.2fx", msSync / max(msAsync, 1e-12)))
}

do {
    try main()
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}

