蘋果晶片架構下減少 GPU Kernel Launch Overhead 之深度研究與優化實務在現代高效能運算（High-Performance Computing, HPC）、即時圖形渲染與大型語言模型（Large Language Models, LLMs）的推論應用中，圖形處理器（GPU）的架構演進已經徹底改變了效能瓶頸的分布。傳統的離散型 GPU 架構高度依賴周邊元件互連高速（PCIe）匯流排進行主機端與設備端的資料傳輸，其效能通常受限於記憶體頻寬。然而，隨著 Apple Silicon 引入統一記憶體架構（Unified Memory Architecture, UMA），中央處理器（CPU）、圖形處理器（GPU）與神經網路引擎（Apple Neural Engine, ANE）得以共享同一塊實體記憶體池 。這種架構消除了傳統的資料複製延遲，使得 LPDDR5 記憶體高達 546 GB/s 的頻寬能被直接利用 。但在消除 PCIe 傳輸瓶頸後，系統效能的限制因素迅速轉移至控制層面，即「核心啟動開銷」（Kernel Launch Overhead）。當運算任務被細分為大量且短暫的微型核心時，CPU 派發任務至 GPU 所耗費的時間，往往遠大於 GPU 實際執行數學運算的週期 。本報告針對 Apple Metal 生態系與 MLX 機器學習框架，進行 exhaustive 的技術剖析，探討如何從底層 API 最佳化、間接指令緩衝區（Indirect Command Buffers, ICB）的應用、Metal Performance Shaders (MPS) 的圖形編譯，乃至於 MLX 框架的延遲評估（Lazy Evaluation）與即時編譯（JIT Compilation）技術，全面且系統性地縮減甚至消除 Kernel Launch Overhead。第一章：Kernel Launch Overhead 之系統底層機制與硬體微架構衝擊1.1 核心啟動開銷的本質、組成與微秒級效能分析Kernel Launch Overhead 是指在實際啟動 GPU 硬體執行緒之前，作業系統與圖形驅動程式必須進行的一系列準備工作。這些工作包含但不僅限於：處理張量形狀等元數據（Metadata）、配置資源綁定、進行記憶體位址轉換、設定管線狀態（Pipeline State），以及最終將指令推入硬體指令佇列中 。在 Apple Silicon 上的 Metal API 底層運作中，這涉及建立 MTLCommandQueue、分配 MTLCommandBuffer，並透過 MTLComputeCommandEncoder 寫入指令 。每一次 API 呼叫都會消耗 CPU 週期，且 GPU 的硬體指令處理器在消化這些控制資料時，亦受限於有限的處理速度 。研究數據指出，在 Apple 的圖形驅動堆疊中，將多個 Metal 指令批次打包至一個 MTLCommandBuffer 提交，其基礎開銷約為 10 微秒（$\mu s$） 。然而，由於這些指令在沒有背景執行緒持續刷新的情況下可能處於閒置狀態，這意味著每一次對編碼器的存取都必須透過互斥鎖（Mutex Lock）進行同步，從而額外增加約 0.2 微秒的延遲 。更甚者，若採用高階的 Metal Performance Shaders Graph (MPSGraph) 框架，雖然其能提供強大的圖形融合功能，但其 API 本身的抽象層處理會進一步增加約 15 微秒的固定啟動開銷 。在極端缺乏最佳化的應用程式中，即使只是將一個完全空白的指令緩衝區推送穿過整個管線，直到 waitUntilCompleted() 方法返回，也可能產生高達 2.5 毫秒的驚人延遲 。1.2 高密度運算與小批次（Small Batch）場景的效能陷阱在大型批次（Large Batch Size）的深度學習訓練任務中，GPU 執行矩陣乘法的時間通常在數毫秒甚至數百毫秒的量級，此時幾十微秒的 CPU 準備時間（CPU Overhead）佔總執行時間的比例微乎其微 。然而，在即時系統（如金融高頻交易或強化學習）、邊緣裝置的音訊處理，以及 LLM 的自迴歸解碼（Auto-regressive Decoding）階段中，系統通常處於批次大小為 1 的極端情境 。在這些小批次場景下，模型推論被分解為大量針對微小張量（Small Tensors）的操作（例如 Element-wise 加法、Softmax、向量仿射變換等）。此時，CPU 的固定派發成本成為決定性因素，導致 GPU 計算單元（ALU）在短暫執行完幾微秒的運算後，便被迫進入閒置狀態，等待下一個指令的到來 。如果不對這些細碎的核心進行融合（Fusion）或採用低延遲的任務佇列策略，系統的整體吞吐量將會崩潰，表現甚至不如高度優化的多核心 CPU 。第二章：Metal 原生 API 架構下的排程、狀態最佳化與記憶體管理要從根本上降低 Apple 裝置上的 Launch Overhead，開發者必須嚴格遵守 Apple 官方文件所建議的 Metal 最佳實務（Metal Best Practices）。Metal 框架的設計初衷即是為了消除早期 OpenGL 的龐大 CPU 狀態機開銷，但這需要開發者在指令生成、資源管理與編譯策略上進行高度配合 。2.1 消除管線停滯：避免同步等待與實現非同步重疊在 Metal 開發中最常見且最具破壞性的效能反模式（Anti-pattern）是過度依賴同步機制。Apple 文件明確警告，開發者絕對不應該在將更多工作排入佇列之前，等待先前的結果完成 。若應用程式呼叫 waitUntilCompleted() 或等待 CPU 核心察覺第一個指令緩衝區完成、開始拆解並觸發回呼（Callback）後才開始編碼下一個緩衝區，將會導致嚴重的管線氣泡（Pipeline Bubbles） 。最佳化策略是實現非同步的 CPU/GPU 重疊執行（Overlap）。系統應在等待第一個指令緩衝區完成的同時，即刻開始編碼後續的指令緩衝區，並持續將其推入佇列。這樣一來，GPU 在完成前一項任務後，能夠無縫接軌地立即處理下一項任務。透過允許 CPU 與 GPU 以此方式進行深度的平行並行運作，整體系統吞吐量可提升高達 10 倍 。2.2 資源池化與狀態切換最小化記憶體分配與編碼器切換在 Metal 中屬於極度昂貴的系統呼叫：資源配置（Resource Allocation）： 動態分配緩衝區（Buffers）與紋理（Textures）會佔用大量的 CPU 週期，導致 CPU 無法穩定地為 GPU 提供運算任務。最佳實務要求盡可能預先分配（Preallocate）並重複使用 MTLResource 物件，或採用三重緩衝（Triple Buffering）等持久化物件策略 。避免編碼器模式切換（Encoder Mode Switching）： 在同一個指令緩衝區中，頻繁在渲染編碼器（Render Encoder）與運算編碼器（Compute Encoder）之間切換，會引發 GPU 內部的硬體模式切換，產生顯著的延遲並破壞吞吐量 。因此，應盡可能將所有運算工作集中批次處理 。減少指令緩衝區數量與資料緊湊化： 由於建立新的指令緩衝區強制要求建立新的編碼器，開發者應將更多的工作塞入更少數量的指令緩衝區中 。同時，保持資料緩衝區的緊湊與結構良好，使用少數大型緩衝區通常比使用大量小型緩衝區更為高效，這能減少描述子（Descriptor）的更新開銷 。2.3 執行緒群組規模化、SIMD 運用與離線編譯為了避免單一運算執行緒受到 Launch Overhead 的限制，必須確保每個執行緒群組（Threadgroup）被分配到足夠的工作量 。開發者應充分利用單指令多資料流（SIMD）操作，將變數群組化為向量（Vectors），以在單一指令中處理多個資料點 。對於需要多個階段處理的影像過濾或卷積運算，將工作分解為區塊（Tiles）並進行多階段串接，可將快取命中率提高，效能提升高達兩倍 。在著色器（Shader）編譯層面，Metal 提供了離線編譯（Offline Compilation）的機制。透過在專案建置時期（Build Time）生成 GPU 二進位檔案，不僅能減少應用程式的啟動時間與關卡載入時的卡頓，還能搭配 MTLLibraryOptimizationLevel 編譯器選項針對二進位檔案大小進行最佳化，大幅降低執行期載入的記憶體與系統匯流排負擔 。第三章：間接指令緩衝區（Indirect Command Buffers）與 GPU 驅動管線革命當 CPU 端的最佳化達到極限，且 API 呼叫本身的成本無法再壓縮時，架構演進的下一步是將指令生成的責任直接轉移至 GPU 本身。Metal 2 引入並在後續版本不斷強化的「間接指令緩衝區」（Indirect Command Buffers, ICB）技術，是徹底消除 CPU 至 GPU 往返延遲（Roundtrip Latency）的核心機制 。3.1 ICB 之持久化與指令重用機制傳統的 MTLCommandBuffer 屬於一次性消耗品（Single-use）。每次執行繪圖或運算任務時，CPU 都必須重新配置所有的頂點緩衝區、幾何變換與繪圖指令 。相較之下，MTLIndirectCommandBuffer 允許開發者將編碼後的 GPU 指令持久化地（Persistently）儲存於系統記憶體中 。在初始化階段，開發者透過 MTLIndirectCommandBufferDescriptor 配置緩衝區的屬性，設定其支援的指令類型（如渲染或運算），並將指令預先編碼。這使得 ICB 代表了一個不透明的資料結構（Opaque Data Structure），可以包含設定獨立頂點資料、轉換矩陣以及網格參數的完整序列 。後續在渲染迴圈中，無論是透過 CPU 或 GPU 的多個執行緒，皆可無限次重複觸發該緩衝區 。這項技術特別適合用於場景中存在大量靜態或結構固定、但參數變動的物件，能將 CPU 端的指令生成開銷降至接近零 。3.2 於 GPU 端編碼指令（GPU-Driven Pipeline）ICB 最強大的潛力在於實現「GPU 驅動渲染與運算」（GPU-Driven Pipeline）。在傳統管線中，若下一步的運算（如繪圖）依賴前一步的運算結果（如視錐體剔除或物理模擬），CPU 必須等待 GPU 回傳結果，進行邏輯判斷後，再生成新的指令發送至 GPU 。這種架構導致了嚴重的管線氣泡與通訊延遲。透過 ICB，開發者可以撰寫一個前置運算核心（Compute Kernel），該核心負責評估邏輯（例如動態識別目前場景中可見的幾何體），並將結果寫入暫存緩衝區。隨後，完全由 GPU 內部觸發後續的 ICB 指令，直接讀取該暫存緩衝區作為輸入，決定要繪製哪些物件，全程無需 CPU 介入 。為了讓運算核心能夠存取 ICB，必須將 ICB 封裝於引數緩衝區（Argument Buffer）中傳遞給核心 。開發者需呼叫 useResource:usage: 或 useHeap:stages: 來宣告 ICB 的使用權限，確保 GPU 在執行前能將相關資源載入暫存器 。在處理 iOS 等行動裝置或 Apple Silicon 平台時，硬體對單次執行核心可存取的資料緩衝區數量有嚴格限制。因此，最佳實務要求採用資料封裝策略：將所有網格（Meshes）資料打包至單一緩衝區的不同偏移量（Offsets）中，並使用另一個專門的緩衝區儲存每個網格的偏移與大小資訊，藉此符合硬體規範並最大化單次 Launch 的處理能力 。3.3 ICB 指令最佳化（Command Optimization）與硬體限制在 GPU 動態生成指令的過程中，特別是涉及條件剔除（Culling）的演算法中，不可避免地會產生許多空指令（Empty Commands）或為不可見物件設置的冗餘狀態（Redundant State） 。
Metal 提供了專門的 API 來進一步減少 ICB 內部的執行期開銷。藉由呼叫 optimizeIndirectCommandBuffer:withRange:，Metal 會對持久化的指令結構進行梳理與壓縮。這能避免硬體指令處理器在執行期浪費寶貴的時鐘週期去跳過這些空白區段，進一步釋放管線效能 。需要注意的是，雖然 ICB 極大地增強了靈活性，但它受到特定硬體架構的限制。例如，在舊款的 A11 仿生晶片（TBDR 架構）上，嘗試將延遲著色（Deferred Shading）的幾何緩衝區傳遞移入 ICB 時，可能會遇到「片段著色器無法與間接指令緩衝區同時使用」的硬體層面錯誤 。此外，根據不同 Metal 版本與硬體家族（如 Metal 3/4, Apple 3/5/9 等），ICB 對記憶體屏障（Memory Barriers）、鑲嵌（Tessellation）與光柵化狀態的支援程度也有所不同，開發者必須參考 Metal Feature Set Tables 進行特化 。比較維度傳統指令編碼 (Traditional Command Buffer)間接指令緩衝區 (Indirect Command Buffer)生命週期與重用單次使用 (Single-use)，需反覆由 CPU 重建 持久化儲存 (Persistent)，可無限次重複執行 生成主體僅限 CPU 端透過 API 序列生成 支援 CPU 多執行緒或 GPU 運算核心並行生成 通訊延遲與依賴性需等待 GPU 完成回傳 CPU，往返延遲高 GPU 內部直接擷取前一 Pass 結果觸發，無 CPU 延遲 資源綁定效率每次 Draw/Dispatch 均需重新綁定 透過 Argument Buffers 傳遞，支援全域資源存取 執行期狀態壓縮依賴 CPU 驅動的狀態追蹤與管理機制支援 optimizeIndirectCommandBuffer 剔除硬體級冗餘狀態 第四章：Metal Performance Shaders (MPS) 與 MPSGraph 編譯層級最佳化對於需要處理複雜張量運算、線性代數與神經網路的領域，直接撰寫 Metal Compute Shaders 開發成本極高。Apple 提供了 Metal Performance Shaders (MPS) 與 MPSGraph 框架，這些框架針對不同世代的 Apple GPU 進行了極致的底層微架構調校 。4.1 符號運算圖（Symbolic Compute Graph）與核心融合機制MPSGraph 的核心優勢在於其高度抽象的圖基（Graph-based）架構。開發者在 MPSGraph 中定義的是一個符號運算圖，其中的操作（如矩陣乘法、卷積、快速傅立葉轉換 FFT）並不立即執行，而是輸出作為邊（Edges）的張量（Tensors） 。這種設計允許 MPSGraph 編譯器在執行前獲取全域視野（Global View），進行徹底的圖形分析與重構。最關鍵的優化是「核心融合」（Kernel Fusion）。在傳統深度學習框架中，連續的卷積（Convolution）、偏差相加（Bias Addition）與啟動函數（Activation）通常需要三次獨立的 Kernel Launch。MPSGraph 能將這三個操作融合為單一高度最佳化的 GPU Kernel 。核心融合不僅減少了兩次 Kernel Launch Overhead，更重要的是消除了中間張量（Intermediate Tensors）往返寫入系統記憶體的龐大頻寬消耗。對於受限於記憶體頻寬（Memory-bound）與激活密集的負載，融合技術能為中小型輸入帶來穩定的 1.5 倍至 3.13 倍速度提升 。4.2 編譯描述子與最佳化層級控制（MPSGraphCompilationDescriptor）在將 MPSGraph 編譯為可執行的 MPSGraphExecutable 時，開發者可以透過 MPSGraphCompilationDescriptor 對編譯器的行為進行精細的控制，以權衡編譯時間、執行期效能與功耗 。編譯器支援不同的最佳化層級（MPSGraphOptimization） ：MPSGraphOptimizationLevel0： 僅執行核心基礎最佳化，編譯速度最快，適合開發調試階段 。MPSGraphOptimizationLevel1（預設值）： 執行進階最佳化，包含配置路徑（Placement Pass）。在此層級下，編譯器會自動分析計算圖的特徵，並將合適的子圖派發至不同的硬體區塊進行異質運算，例如協同運用 GPU 與功耗效率極高的神經引擎（Apple Neural Engine, ANE）以及 CPU 。此外，開發者可設定 reducedPrecisionFastMath 屬性，允許跨整個執行檔採用降低精度的快速數學運算（如強制轉換為 FP16） 。同時，利用 MPSGraphOptimizationProfile 枚舉，能設定啟發式策略（Heuristics），指示編譯器以極致效能（performance）或功耗效率（powerEfficiency）為導向進行網路最佳化 。4.3 效能極限與 API 開銷的權衡儘管 MPSGraph 提供了優異的圖層級最佳化，但其作為高階抽象框架，不可避免地會帶來一定的系統負擔。效能剖析指出，MPSGraph 分配了大量的內部中間資源，開發者必須謹慎管理其記憶體消耗 。此外，正如第一章所述，MPSGraph API 本身會產生約 15 微秒的額外延遲 。因此，在極端追求低延遲且操作過於細碎的場景中，若發現 MPSGraph 無法有效融合特定節點，開發者應考慮退回使用底層的原始 MPS 核心，甚至透過 MPSGraph 的自訂函數支援，直接撰寫自訂的 Metal Shading Language (MSL) 著色器來避開抽象層帶來的固定成本 。第五章：MLX 框架之架構革新與延遲評估（Lazy Evaluation）為了解決傳統機器學習框架（如 PyTorch）在 Apple Silicon 上的水土不服，Apple 機器學習研究團隊推出了原生陣列運算框架：MLX 。相較於將 CUDA 邏輯強行映射至 Metal 的方法，MLX 的底層邏輯完全圍繞著統一記憶體架構與極低延遲派發構建 。5.1 統一記憶體與資料傳輸的徹底消除在 PyTorch 或 TensorFlow 中，GPU 與 CPU 被視為獨立的設備，資料交換需要透過 PCIe 匯流排。MLX 充分利用了 Apple Silicon 共享物理記憶體的優勢。在 MLX 中，陣列（Arrays）直接駐留於統一記憶體中，運算可在 CPU 或 GPU 上無縫切換，而無需進行任何資料傳輸（Zero-copy access） 。這項特性直接消滅了主機與設備間的同步屏障（Synchronization Barriers），使得框架設計能夠將焦點完全集中在優化運算派發本身的效率上 。5.2 延遲評估（Lazy Evaluation）與動態運算圖MLX 的運作基礎是「延遲評估」機制。當開發者在 Python 中呼叫 MLX 的數學運算時，系統實際上並未立即觸發硬體執行，而是動態地記錄並構建一個計算圖（Compute Graph） 。真正的硬體運算只有在明確呼叫 mx.eval() 時，或是陣列資料被實際存取（如列印、轉換為 NumPy 陣列或用於控制流 if y > 0:）時才會發生 。這種機制帶來了巨大的效能優勢：消滅未使用的運算與智慧裁剪： 由於圖形是在執行前構建，MLX 分析器會自動裁剪未被最終結果引用的運算節點，節省了無謂的派發開銷 。記憶體峰值壓縮： 例如在載入龐大的 mlx.nn.Module 大型語言模型時，可以先以延遲狀態初始化模型結構，隨後直接載入 float16 權重，這避免了 Eager 模式下預設 float32 初始化所帶來的高達兩倍的峰值記憶體消耗 。攤銷派發成本與減少固定 Overhead： 由於每次調用 mx.eval() 都會產生固定的架構開銷，開發者可採取最佳策略：將 eval() 放置於訓練迴圈的外部，或推論生成階段的特定檢查點。排程器會累積成百上千個操作，在一次評估中將其打包派發至 GPU，大幅減少框架層面的 Python 迴圈與底層溝通開銷 。5.3 核心技術：mx.compile 與 JIT 融合（Just-In-Time Fusion）為了將 GPU Kernel Launch Overhead 降至硬體極限，MLX 提供了強大的即時編譯（JIT）裝飾器：@mx.compile 。
當一個純函數被標記為 mx.compile 時，MLX 框架會對其進行追蹤（Trace），建立最佳化的運算圖，並將其編譯為單一的 Metal 核心 。常數折疊（Constant Folding）與操作融合： 編譯器在編譯期會預先計算所有常數操作。更重要的是，它會識別出可合併的操作序列（如 $a * b + c$），將其替換為等效且更高效的單一融合乘加（FMA）指令核心 。這不僅減少了 Kernel 數量，更將中間變數保留在 GPU 暫存器或 L2 快取中，大幅降低 DRAM 寫入負擔 。動態形狀（Dynamic Shapes）與快取機制： 傳統框架的靜態編譯在改變輸入張量形狀時，往往會引發極度緩慢的重新編譯。MLX 的圖構建是完全動態的，其基於模板（Template-based）的 Metal 編譯器能妥善處理變動的形狀，並將編譯後的 .metallib 二進位檔快取於記憶體中，供後續步驟無縫重複使用 。優化機制Eager Evaluation (急切評估)MLX Lazy Evaluation + mx.compile圖構建方式逐行獨立執行，無全域視野延遲記錄，建立完整 DAG 進行全域最佳化 核心派發次數極高（每個微小算子皆觸發一次 Launch）極低（透過融合坍縮成單一或少數 Kernel） 中間張量記憶體存取頻繁將中間結果讀寫至主記憶體中間變數駐留於 GPU 暫存器，無全矩陣寫入 DRAM Python Overhead 影響嚴重受限於 GIL 與直譯器延遲完全剝離，執行緒交由編譯後的 Metal 二進位檔接管 5.4 排程器優化與自訂基元（Custom Primitives）的深層考量除了編譯層，MLX 的底層硬體排程器（Scheduler）同樣針對 Launch Overhead 進行了特化。為了減輕小核心頻繁啟動的負擔，排程器會執行「指令緩衝區群組化」（Command Buffer Grouping），自動將多個小型操作打包進單一個 Metal 指令緩衝區中統一提交 。同時，透過維護執行期的動態依賴圖，排程器能精準識別無相互依賴的操作，將其派發至不同的硬體串流（Streams）中進行平行的硬體執行，達成運算資源的極致利用 。然而，當研究人員需要實作極度特化的演算法（例如未被框架內建涵蓋的自訂量化稀疏矩陣運算，或是 PagedAttention），且希望維持延遲評估系統的無縫整合時，開發者會面臨架構選擇。最簡單的方式是使用 mx.metal.custom_kernel，這也是官方建議的首選途徑 。但如果自訂核心的設計不當，可能會強制要求對輸入張量進行顯式的 x.eval()，從而破壞了計算圖的連續性並引入了同步屏障（Synchronization Barriers） 。為了解決此問題，進階開發者會將自訂 C++ 運算註冊為 MLX 的底層基元（Primitive），這樣不僅能完全融入 Lazy Evaluation 系統，還能讓排程器統一管理其執行緒池（Threadpool）與跨設備串流調度，確保在單一計算圖中完成所有運算，徹底消除每層之間（Per-layer）的同步等待 。第六章：PyTorch MPS 後端與 MLX 架構之對比分析為了更深刻理解 MLX 在 Apple Silicon 上的優勢，我們必須探討 PyTorch 在相同硬體上的運作機制及其面臨的挑戰。6.1 torch.compile 與 CUDA Graphs 的局限性在 PyTorch 2.0 中，為了減少 Python 與 Kernel Launch Overhead，官方推出了強大的 torch.compile 編譯器 。透過設定 mode="reduce-overhead"，PyTorch 會利用 CUDA Graphs 捕捉靜態的運算圖，將一系列核心啟動紀錄下來，並在後續疊代中直接重播（Replay）這個靜態圖，從而大幅減少 CPU 的介入 。然而，這種架構存在根本的局限。首先，CUDA Graphs 的捕捉機制強烈依賴靜態形狀。一旦輸入的 Batch Size 或序列長度改變，系統就必須觸發昂貴的重新編譯，這在動態生成的 LLM 應用中是致命的 。相較之下，MLX 的 mx.compile 處理變動形狀的能力更強，無需針對每個 Batch Size 建立獨立的捕捉迴圈 。其次，PyTorch 的架構預設主機（CPU）與設備（GPU）是分離的，即使在 Apple Silicon 上使用了 MPS 後端，其內部架構仍殘留了大量為了處理 PCIe 傳輸而設計的冗餘機制，無法像 MLX 那樣純粹地依賴統一記憶體達成 Zero-copy 運作 。6.2 PyTorch MPS 後端之效能瓶頸根據近期的效能評估，PyTorch 的 MPS 後端在處理特定的硬體加速操作時面臨挑戰。例如，Apple Silicon 目前缺乏對部分進階 FP64 操作的原生支援，導致依賴這些操作的科學運算函式庫（如 PyRadiomics）在 MPS 上的效能不如預期，甚至需要退回純 CPU 或 CUDA 環境執行 。此外，在處理小型卷積或非標準的運算（如特定步長的 MaxPool2D）時，PyTorch 的自動求導（Autograd）與啟動開銷極大，社群甚至需要透過演算法層面的替代方案，將大型 Kernel Size 拆分為多個小型 Kernel，以降低每個單元的計算成本與 Backpropagation Overhead 。這些現象突顯了在非原生框架上進行底層優化所面臨的結構性阻力。第七章：2026 年突破性研究：LLM 推論與科學運算之效能實證減少 Kernel Launch Overhead 的最終目的，在於釋放 Apple Silicon 潛藏的強大運算頻寬與 ALUs 能力。以下透過 2026 年最新的研究數據，展現這些架構最佳化在實際應用中所帶來的顛覆性突破。7.1 大型語言模型與多模態推論之效能飛躍在 LLM 推論的自迴歸生成（Auto-regressive Generation）階段，模型必須逐字元（Token-by-token）生成輸出。這導致每一層 Transformer Block 的運算皆是極小批次的矩陣乘向量（GEMV）操作，這正是 Kernel Launch Overhead 殺傷力最大的場景 。根據 2026 年發表於 EuroMLSys 的論文《Native LLM and MLLM Inference at Scale on Apple Silicon》（Barrios 2026），研究團隊透過整合 MLX 框架開發了 vLLM-mlx 引擎 。該引擎利用 mx.compile 將 Forward Pass 中的注意力機制（Attention）、多層感知機（MLP）與啟動函數深度融合為單一 Metal 核心，成功跨越了基礎移植的效能瓶頸 。配合量化 KV Cache 與前綴共享（Prefix Sharing）技術，系統能更有效利用有限的 LPDDR5 記憶體頻寬 。實測數據展現了驚人的結果：在配備 M4 Max 晶片與 64GB 記憶體的平台上，針對 Qwen3-VL-4B 等多模態模型，mx.compile 的融合結合視覺內容快取（Vision Caching）技術，將高達 1024x1024 解析度影像的編碼延遲從 21.7 秒巨幅縮減至 1.15 秒以下，達成了高達 19 倍的加速比 。在純文字生成上，有效吞吐量更攀升至 525 tokens/second 。快取機制 (Caching Configuration)解析度 / 負載類型基準延遲 (Baseline Latency)優化後延遲 (Optimized)速度提升 (Speedup)無快取 (No caching)Qwen3-VL-4B21.7s-1.0x 僅 KV 快取 (KV cache only)Qwen3-VL-4B-18.2s1.2x 僅視覺嵌入 (Vision embeddings)Qwen3-VL-4B-2.8s7.8x 完整快取 + 編譯融合 (Full cache)Qwen3-VL-4B21.7s1.15s19.0x 影片分析 (64 Frames @ 8fps)Qwen3-VL-4B-18.2s (Cached)24.7x 在與業界標準 llama.cpp 的效能對比中，雖然 llama.cpp 擁有極度優化的手刻 C++ Metal Kernel 與 GGUF 格式 ，但 MLX 透過架構級別的編譯與優化，在多輪對話與動態代理（Agentic Workflows）負載下，其 FP16 精度的生成速度已與 llama.cpp 並駕齊驅（例如 Qwen3 30B 模型下，兩者皆穩定於 55-58 tokens/s 區間） 。這證明了 MLX 的 JIT 融合技術能夠在不需要繁重的人工組合語言調優下，達到甚至超越系統級底層框架的吞吐量 。7.2 降維演算法與高密度音訊處理之突破Kernel Launch Overhead 的消除不僅惠及大型語言模型，在傳統科學運算與高密度訊號處理中同樣展現了強大的威力。在處理維度約簡（Dimensionality Reduction）如 t-SNE、UMAP 與 PaCMAP 時，演算法中包含海量的隨機梯度下降（SGD）更新與排斥力核心計算。在 2026 年的 mlx-vis 專案研究中，開發者完全捨棄了傳統的 CPU 實作（如 sklearn 或 Cython），將核心運算全部移植至 MLX 。透過將最耗時的內部迴圈標記為 @mx.compile，MLX 編譯器成功將數萬次的算子派發塌陷（Collapse）為少數幾個巨大的融合核心，徹底抹除了 Python 直譯器與 API 呼叫的開銷 。實測顯示，在處理 70,000 個資料點（Fashion-MNIST）時，UMAP 的執行時間從傳統的 8.52 秒縮短至 3.23 秒（2.6 倍加速）；t-SNE 更是從 58.62 秒暴跌至 3.78 秒（15.5 倍加速） 。降維演算法 (Method)迭代次數 (Iters)傳統 CPU 參考實作 (Reference)MLX-vis GPU 編譯實作速度提升 (Speedup)UMAP5008.52 ± 1.91 s (umap-learn)3.23 ± 0.02 s2.6x t-SNE50058.62 ± 0.96 s (openTSNE)3.78 ± 0.02 s15.5x PaCMAP4506.56 ± 0.04 s (pacmap)2.13 ± 0.04 s3.1x TriMap50014.32 ± 0.06 s (trimap)2.40 ± 0.07 s6.0x 在即時音訊處理領域，以 Silero 語音活動檢測（VAD v6）為例，研究指出在使用純 MLX 框架進行串流處理時，若未經編譯的「Naive GPU」模式，其每區塊處理延遲為 1.46 毫秒 。然而，當套用 mx.compile 並結合 CPU 串流排程後，延遲瞬間減半至 0.75 毫秒。在最新的 M5 Max 晶片上，透過非同步批次處理（Async-batched）與 mx.eval 的屏障縮減（Barrier Reduction）技術攤銷派發成本，每區塊吞吐量進一步壓縮至驚人的 0.17 毫秒，達成了 187 倍的即時處理速度（Real-time ratio），將純 MLX 實作與硬體神經引擎（ANE）特化的 CoreML 之間的效能差距縮小了 50% 以上 。第八章：結論與前瞻性技術展望綜上所述，在 Apple Silicon 獨特的統一記憶體架構下，解決 GPU Kernel Launch Overhead 已經超越了單純的程式碼最佳化範疇，成為決定系統整體吞吐量、記憶體效率與應用程式可行性的最核心課題。從底層硬體指令的微秒級分析到高階機器學習框架的架構重構，技術演進的軌跡清晰地表明，效能優化的重心正全面向著「編譯期全域優化」與「GPU 自治驅動」轉移。從 Metal API 的演進來看，開發者必須徹底揚棄傳統的 CPU/GPU 同步等待思維，轉而擁抱深度管線重疊（Pipeline Overlap）與資源持久化池化策略。間接指令緩衝區（ICB）的成熟，賦予了 GPU 動態調度與自主生成指令的能力，這徹底切斷了依賴 CPU 進行小任務派發的鎖鏈，將 API 呼叫的沉重負擔從關鍵路徑中移除。在抽象運算與神經網路層面，高階封裝如 MPSGraph 透過符號運算圖的建構，允許系統在底層自動執行硬體單元分配與算子融合；而 MLX 框架的誕生更是標誌著 Apple 生態系專屬高效能運算的里程碑。MLX 的核心哲學——「延遲評估（Lazy Evaluation）」與「即時編譯（mx.compile）」——完美取代了傳統機器學習框架複雜且僵化的圖形捕捉機制。透過將無數細碎的張量操作塌陷、融合為龐大而單一的 Metal Kernel，結合排程器的指令緩衝區群組化與無鎖並行機制，MLX 成功在 LLM 推論、降維科學運算與即時音訊分析等極端考驗 Launch Overhead 的場景中，達成了十倍甚至數十倍的效能飛躍。未來的 Apple Silicon 高效能應用，無論是追求極致吞吐量的邊緣人工智慧代理（Agentic Workflows）、即時混合實境渲染，還是大規模物理模擬，都必須深度整合並內化這些架構最佳化原則。唯有透過盡可能將運算邏輯融合、利用排程器將指令平行打包，並將控制流的決策權下放至 GPU 本身，系統架構師方能真正克服啟動開銷的物理極限，毫無保留地釋放硬體前所未有的龐大算力。

蘋果晶片環境下 Metal 運算核心排程與機器學習計算圖排程最佳化研究報告緒論：核心啟動開銷在現代運算框架中的效能瓶頸在高效能運算與硬體加速機器學習領域中，圖形處理單元 (GPU) 的理論峰值效能往往無法在實際應用中完全轉化為真實的吞吐量。無論是科學運算或是深度神經網路的推論與訓練，系統層級的效能瓶頸通常不在於 GPU 的浮點運算能力，而在於主機端 (Host-side) 中央處理器 (CPU) 的排程編排，特別是 GPU 運算核心啟動開銷 (Kernel Launch Overhead) 。GPU 運算的延遲模型可以被解構為三個主要部分：主機端的指令編碼與發送開銷、硬體層級的佇列同步延遲，以及 GPU 運算核心的實際執行時間。現代深度學習架構，例如大型語言模型 (Large Language Models, LLM) 與混合專家模型 (Mixture of Experts, MoE)，在處理單一詞元 (Token) 時，需要執行數以千計的高度細粒度 (Granular) 張量運算 。當這些微小且密集的運算透過傳統的即時執行 (Eager Execution) 模式被序列化地發送至 GPU 時，啟動開銷便會佔據整體執行時間的絕大比例。一項針對序列吞吐量效能的數學分析精確地展示了此問題的嚴重性。假設在一個循環次數高達一萬次的訓練或推論迴圈中，包含一次矩陣乘法與五次逐元素 (Pointwise) 運算。若 CPU 執行矩陣乘法需時 100 微秒，逐元素運算各需 1 微秒，則 CPU 的單次迴圈總時間為 105 微秒。假設 GPU 的浮點運算能力與記憶體頻寬分別為 CPU 的 100 倍與 10 倍，矩陣乘法的硬體執行時間將降至 1 微秒，而逐元素運算則降至 0.1 微秒。然而，若機器學習框架在每次驅動 GPU 核心時產生 50 微秒的排程開銷，則 GPU 的單次迴圈總耗時將激增至 301.5 微秒。在這種情況下，擁有強大算力的 GPU 系統，其最終執行速度反而比純 CPU 系統慢上近三倍 。在 Apple Silicon 環境下（例如配備 19 核心 GPU 與高達 200 GB/s 記憶體頻寬的 M2 Pro 晶片 ），解決此一瓶頸需要結合底層硬體特性分析、Metal API 的進階指令緩衝區 (Command Buffer) 資源管理、中介碼 (Intermediate Representation) 編譯器指令優化，以及如 Apple MLX 框架所提供的高階計算圖 (Computational Graph) 排程機制。本報告將深入探討並統整這些層面的最佳實踐與實作方案。學術界與工業界對 GPU 核心啟動開銷的探索與研究學術界針對 GPU 核心啟動開銷的優化已經累積了大量研究，這些研究主要聚焦於如何縮短 CPU 的等待時間以及最大化 GPU 的運算重疊 (Overlap)。在分散式系統與多 GPU 叢集上的大型語言模型推論研究中，學者發現多 GPU 系統效能低落的原因往往不是 GPU 資源飽和，而是 CPU 無法足夠快速地餵給 GPU 運算指令。在 CPU 資源受限的配置下，系統會出現核心啟動延遲、通訊停滯以及詞元化延遲等症狀，導致嚴重的 GPU 閒置 。研究指出，若能確保充足的 CPU 資源或優化 CPU 端的排程邏輯，可以在不增加 GPU 資源的情況下，將首個詞元生成時間 (Time-to-First-Token, TTFT) 的延遲降低 1.36 倍至 5.40 倍 。為了克服這些挑戰，學術界提出了多種編譯器與排程優化框架。例如，EvoEngineer 框架透過大型語言模型自動生成並演化 CUDA 與硬體核心程式碼，在兼顧正確性的前提下，成功使基準測試中的核心程式碼獲得高達 36.75 倍的加速，並在超過一半的測試中實現兩倍以上的效能提升 。另一個名為 AutoGraph 的統一框架，則採用動態規劃與回溯搜尋演算法，針對深度神經網路的計算圖進行最佳化，透過分析關鍵路徑的混合成本，獲得有利於 GPU 核心平行執行的計算圖結構，相較於現有的計算圖最佳化方法，可提升 3.47 倍的執行速度 。在針對邊緣裝置與行動 GPU 的研究中，學者分析了卷積神經網路 (CNN) 在行動平台上的推論時間，並明確指出 GPU 核心啟動開銷是極為顯著的效能瓶頸。透過建立效能模型來預測並最佳化核心刷新 (Kernel Flush) 的週期，研究人員在 Adreno 與 Mali 系列 GPU 上使用 TensorFlow Lite 等框架，成功實現了高達 64% 的推論加速 。此外，Event Tensor 這一種為動態巨型核心 (Dynamic Megakernel) 設計的統一編譯器抽象層，透過將多個運算子融合為單一的持續性核心 (Persistent Kernel)，徹底消除了核心之間的啟動間隙 (Launch Gaps)，並將圖塊化 (Tiled) 任務的依賴關係進行編碼，為大型語言模型服務提供了極低的延遲與預熱開銷 。這些研究共同指明了一個明確的技術發展方向：必須盡可能地將離散的張量運算融合 (Fusion)，減少主機端的 API 呼叫次數，並將排程控制權從 CPU 轉移至 GPU 端，或透過靜態分析在編譯期即決定最佳的執行路徑。Apple Silicon 統一記憶體架構 (UMA) 與底層硬體特性分析要徹底解決 Apple 平台上的核心啟動開銷，必須深刻理解 Apple Silicon 的實體架構。傳統的離散型 GPU (Discrete GPU) 依賴 PCIe 匯流排與主機記憶體進行通訊，這導致了昂貴的記憶體複製 (Memory Copy) 成本。而 Apple 的 M 系列晶片（如 M1、M2 Pro、M3 Max 等）採用了統一記憶體架構 (Unified Memory Architecture, UMA)，將高頻寬的 LPDDR 記憶體模組與 CPU 及 GPU 進行實體共享 。這種架構的轉變使得顯式的設備到主機 (Device-to-Host) 資料傳輸變得毫無必要。若機器學習框架未能識別 UMA 的優勢，仍以傳統的記憶體管理模式運作，將會在無形中產生巨大的系統負擔。例如，在 llama.cpp 早期的 Metal 後端實作中，就曾因為不必要的記憶體傳輸開銷而受到效能限制，而採用零複製 (Zero-Copy) 張量運算的框架則能顯著提升吞吐量 。Apple Silicon 的 GPU 擁有獨特的記憶體層級與執行模型，這直接影響了開發者在使用 Metal Shading Language (MSL) 撰寫自訂運算核心時的最佳化策略：暫存器 (Registers)： 位於算術邏輯單元 (ALU) 旁的最快速儲存空間。在高效能的矩陣運算中，將資料載入暫存器後進行展開與運算，是達到硬體理論峰值的關鍵 。執行緒群組記憶體 (Threadgroup Memory)： 相當於 NVIDIA 架構中的共享記憶體 (Shared Memory)。Apple Silicon 對於每個執行緒群組施加了嚴格的 32 KiB 記憶體限制 。在進行快速傅立葉轉換 (FFT) 或大型矩陣乘法時，若工作負載超出此限制，便需要將演算法拆分為多個階段。研究指出，Apple GPU 在執行緒群組記憶體的同步屏障 (Barrier) 上具有極低的延遲（約兩條指令週期），這鼓勵開發者積極使用執行緒群組層級的資料交換 。系統記憶體 (Device Memory)： 即共享的 UMA 空間。在硬體執行模型上，Apple GPU 將執行緒劃分為 SIMD 群組 (SIMD-groups)，通常包含 32 個執行緒。硬體內部具備專門用於矩陣乘加 (Matrix Multiply Accumulate, MMA) 的協同處理器。透過 metal::simdgroup*matrix 函式庫，開發者可以指示硬體從執行緒群組記憶體中協同載入 8x8 的矩陣片段，並在不需要透過暫存器進行中介資料傳輸的情況下，直接進行同步的乘加運算 。此外，Apple GPU 核心在設計上對緩衝區綁定 (Buffer Binding) 的數量設有物理限制。研究人員指出，單一 Metal 運算管線最多僅能直接綁定 31 個緩衝區插槽 (Slots)。當深度學習模型需要傳遞超過此數量的張量或參數時，框架必須退而求其次，配置一個額外的緩衝區來儲存指標位址，這種技術被稱為統一共享記憶體 (Uniform Shared Memory, USM) 指標。GPU 必須先將這些 USM 指標載入暫存器中進行數值複製，隨後再進行解參考 (Dereference)，這會增加編譯器的邏輯複雜度與記憶體讀取延遲 。因此，盡可能減少緩衝區的綁定數量，並將相關參數封裝，是降低排程成本的硬體級法則。Metal API 的低階排程最佳化與指令緩衝區策略Metal 框架被設計為提供最低開銷的 GPU 存取途徑 。然而，若未遵循最佳實踐，開發者仍可能在不知不覺中累積龐大的 CPU 端開銷。標準的 Metal 運算排程流程包含：從 MTLCommandQueue 獲取 MTLCommandBuffer，實例化 MTLComputeCommandEncoder，透過 setComputePipelineState 設定管線狀態，使用 setBuffer:offset:atIndex: 綁定記憶體資源，並最終呼叫 dispatchThreads:threadsPerThreadgroup: 派發運算任務 。上述每一個 API 呼叫都需要跨越軟體邊界，進入 Metal 驅動程式進行資源驗證與記憶體依賴性分析。為了消除這些系統層級的開銷，進階應用程式與機器學習框架採用了以下幾種核心策略：資源危險追蹤解除與非保留引用預設情況下，Metal 的指令緩衝區會對所有綁定的資源建立強引用 (Strong References)，確保資源在 GPU 執行完成前不會被作業系統回收。同時，Metal 會自動追蹤資源的讀寫危險 (Hazard Tracking)，並在必要時插入隱式的記憶體屏障，以防止資料競爭 (Data Race)。這種自動化管理雖然安全，卻會大幅消耗 CPU 資源。為追求極致效能，開發者會在建立 MTLBuffer 或 MTLTexture 時，明確指定 MTLResourceHazardTrackingModeUntracked 屬性。這會向驅動程式宣告，應用程式將自行負責管理所有讀寫依賴，驅動程式無需進行任何追蹤 。此外，透過呼叫指令佇列的 makeCommandBufferWithUnretainedReferences() 建立指令緩衝區，Metal 將不再對引用的資源進行保留計數 (Retain Count) 追蹤 。然而，這種優化策略伴隨著極高的風險。由於資源生命週期完全由開發者控管，若 CPU 執行緒在指令緩衝區仍在 GPU 飛行 (In-flight) 階段時，錯誤地釋放了相關的記憶體資源，將導致嚴重的系統崩潰 (Kernel Panic)。在開源社群的除錯紀錄中，就曾出現由於並行自訂核心與未追蹤資源衝突，導致 IOGPUMemory.cpp:550 拋出 completeMemory() prepare count underflow 錯誤的案例，這通常是由於無限增長的 KV Cache 被作業系統強制回收所引起 。因此，採用此策略的程式碼架構必須擁有極為嚴謹的自訂記憶體配置器 (Allocator)。引數緩衝區 (Argument Buffers) 的應用為了解決動輒數十次的 setBuffer 與 setTexture 綁定呼叫所累積的編碼成本，Metal 提供了引數緩衝區 (Argument Buffers) 機制 。透過 MTLArgumentEncoder，開發者可以在應用程式初始化或模型載入階段，將多個緩衝區、紋理、甚至常數資料，打包編碼進單一的 MTLBuffer 中 。在關鍵的運算迴圈中，CPU 僅需要對 MTLComputeCommandEncoder 呼叫一次 setBuffer，將預先打包好的引數緩衝區綁定至指定的索引位置。這種做法將 $O(N)$ 的綁定複雜度壓縮至 $O(1)$，不僅避開了硬體 31 個綁定插槽的限制，更將龐大的 CPU 編碼開銷從關鍵渲染或運算路徑 (Critical Path) 轉移至初始化階段，顯著提升了框架處理大型深度學習模型時的效率 。Metal 4 核心 API 的異步與解耦機制Apple 在 Metal 4 中引入了全新的 MTL4CommandQueue 協定，進一步優化了指令緩衝區的生命週期與記憶體開銷。在先前的架構中，MTLCommandBuffer 必須由特定的指令佇列產生，且其提交 (Commit) 方法是綁定的。Metal 4 改變了這一點，開發者可以直接透過 MTLDevice 的工廠方法（例如 makeCommandBuffer()）建立指令緩衝區，並可以將其提交給屬於同一裝置的任何 MTL4CommandQueue 實例 。更重要的是，Metal 4 允許開發者透過 commit:count: 方法，將多個指令緩衝區作為一個群組同時提交。這種解耦機制使得應用程式能夠在不同的背景工作執行緒 (Worker Threads) 上平行編碼多個指令緩衝區，隨後統一提交至 GPU，最大化 CPU 核心的利用率，同時確保 GPU 佇列持續飽和 。優化技術名稱實作機制與 API 呼叫主要效益風險與注意事項無追蹤資源管理MTLResourceHazardTrackingModeUntrackedmakeCommandBufferWithUnretainedReferences()消除 Metal 驅動程式層級的保留計數與危險追蹤開銷。若生命週期管理不當，將導致 IOGPUMemory.cpp 崩潰與 Kernel Panic 。引數緩衝區 (Argument Buffers)MTLArgumentEncoder將多個資源指標打包入單一 MTLBuffer將 O(N) 的 API 綁定成本降至 O(1)，規避硬體插槽限制 。需在初始化階段進行正確的指標映射。Metal 4 解耦佇列MTL4CommandQueuedevice.makeCommandBuffer()支援跨執行緒平行編碼，透過群組提交減少排程延遲 。需自行管理跨執行緒的資源狀態同步。間接指令緩衝區 (Indirect Command Buffers, ICB) 的架構與實作在所有 Metal API 中，減少 CPU 參與度最激進且最有效的機制，莫過於間接指令緩衝區 (Indirect Command Buffers, ICB) 。傳統上，指令緩衝區在提交給 GPU 並執行完畢後便會被銷毀，若要在下一幀或下一次推論迭代中執行相同的運算，CPU 必須重新建立並編碼所有的指令。ICB 透過 MTLIndirectCommandBuffer 類別，允許開發者將 GPU 指令持久化地儲存於記憶體中，並可被重複使用 。更具突破性的是，ICB 開啟了 GPU 驅動管線 (GPU-Driven Pipeline) 的可能性。開發者可以撰寫一個負責邏輯判斷的運算核心，由該核心直接在 GPU 上動態產生並編碼後續的運算或渲染指令。這不僅徹底移除了 CPU 與 GPU 之間的通訊橋樑延遲，更實現了無縫的運算重疊 。ICB 的配置與派發邏輯建立 ICB 的第一步是配置 MTLIndirectCommandBufferDescriptor。開發者必須預先宣告該緩衝區的最大指令數量，以及允許編碼的指令類型（例如 .concurrentDispatch 或 .concurrentDispatchThreads）。接著透過裝置實例的 makeIndirectCommandBuffer(descriptor:maxCommandCount:options:) 方法配置實體記憶體 。在 Swift 層級，若要透過 CPU 對 ICB 進行編碼，其流程與標準編碼器略有不同 ：呼叫 indirectComputeCommandAt(*:) 從 ICB 獲取一個特定索引的 MTLIndirectComputeCommand 實例。透過 setComputePipelineState(_:) 為該指令設置執行管線。使用 setKernelBuffer(_:offset:at:) 綁定所需的記憶體緩衝區。設定執行緒群組配置，如 setThreadgroupMemoryLength(_:index:)。呼叫 concurrentDispatchThreads(_:threadsPerThreadgroup:) 將執行網格幾何資訊編碼入指令中。當所有指令編碼完成後，應用程式並非直接提交 ICB，而是需要建立一個常規的 MTLComputeCommandEncoder，並呼叫其 executeCommandsInBuffer(_:range:) 方法。這會指示 Metal 驅動程式去執行 ICB 內部指定範圍內的指令集合 。需要特別注意的是，因為 executeCommandsInBuffer(_:range:) 本身並不隸屬於任何特定的管線階段 (Stage)，開發者無法在外部使用常規的記憶體屏障 (Barrier) 來等待整個 ICB 的完成。然而，ICB 內部編碼的每一個獨立指令，仍會嚴格遵循與直接編碼相同的階段同步規則 。ICB 的編譯層級最佳化動態產生或多次重複使用的 ICB 內部，不可避免地會產生冗餘狀態。例如，相鄰的多個指令可能設定了完全相同的管線狀態，或者某些條件判斷導致產生了空的執行指令。為了解決這個問題，Metal 提供了 optimizeIndirectCommandBuffer(\_:range:) 方法，這是一個透過 MTLBlitCommandEncoder 執行的硬體級最佳化指令 。此方法會命令 GPU 掃描 ICB 中指定的指令範圍，移除空白指令並消除重複的狀態設定。這種緊湊化 (Compaction) 過程能夠釋放大量的記憶體空間，並減少硬體排程單元在執行階段所需解析的狀態轉換次數，進一步降低延遲 。然而，這種最佳化帶來了嚴格的執行時限制：一旦 ICB 的某個範圍被最佳化，主機端呼叫 executeCommandsInBuffer 時，其指定的執行範圍必須完全涵蓋該最佳化區段的起點，且結束於區段內部或末端。開發者絕對不能嘗試從最佳化區段的中間索引開始執行，否則將導致未定義的硬體行為 (Undefined Behavior) 。組合語言 (ASM) 限制與 Metal Shading Language (MSL) 底層指令最佳化機器學習框架的效能往往取決於對底層硬體指令集架構 (ISA) 的精確控制。Apple Silicon 的 GPU 包含未公開的矩陣乘加 (MMA) 硬體加速單元。過去，為突破編譯器的抽象層，追求極致效能的開發者與研究人員（如 Asahi Linux 專案與效能評測專家 Philip Turner）曾廣泛使用內聯組合語言 (Inline Assembly) 技巧 。在 MSL 檔案中，開發者曾利用 **asm 指令直接呼叫 Apple Intermediate Representation (AIR) 的特殊函式。例如，透過撰寫 **asm("air.simdgroup_async_copy_1d.p3i8.p1i8")，開發者可以強制 GPU 執行特定的記憶體非同步複製或高效能 SIMD 歸約 (Reduction) 運算 。這種技術不僅被用於矩陣運算，也被應用於實現如 Unreal Engine 5 Nanite 技術所需的 64 位元原子比較 (Atomic Comparisons) 操作中 。**asm 關鍵字的禁用與過渡然而，Apple 在其近期的開發者工具更新中（如 Xcode 16 與 macOS Tahoe Beta 4）實施了嚴格的安全性與編譯器政策，全面禁用了 Metal 著色器中的 **asm 關鍵字。嘗試編譯包含該指令的程式碼會觸發 illegal string literal in 'asm' 的編譯錯誤 。這項改變迫使依賴未公開指令的機器學習框架必須重新架構其底層實作。原生硬體加速指令與原子操作的應用為取代內聯組合語言，開發者必須轉向 Apple 官方提供的原生 C++ 封裝 API。在 MSL 中，metal::simdgroup_matrix 命名空間提供了存取硬體 MMA 單元的合法途徑。透過這些原生函式，執行緒群組可以協同地從記憶體載入 8x8 的矩陣片段，並執行同步的乘加運算，而無需經過繁瑣的暫存器資料交換 。此外，針對高頻寬的平行累加操作（例如神經網路反向傳播中的梯度累積），開發者應充分利用 Metal 支援的原子操作 (Atomic Operations)。透過在核函式宣告中使用 device atomic_float* 或針對 Apple Silicon 特別優化的 64 位元原子類型 atomic_ulong，並結合 atomic_fetch_add_explicit 等函式，可以在不需要引入低效的執行緒群組屏障 (Threadgroup Memory Barriers) 的情況下，安全地進行記憶體寫入 。放棄 \_\_asm 而改用原生 MSL 指令的另一個隱藏優勢在於編譯器最佳化。當使用不透明的組合語言字串時，LLVM 編譯器無法理解其內部邏輯，進而放棄許多常規的最佳化通道（如迴圈展開 Loop Unrolling 或暫存器重新分配）。使用原生函式則允許 Metal 編譯器全面分析計算圖，產生更高效的機器碼 。Metal Performance Shaders (MPS) Graph 的編譯層級最佳化對於不需撰寫自訂核心的開發者，Apple 提供了 Metal Performance Shaders (MPS) 框架，其中的 MPSGraph 是一個強大的多維度計算圖引擎 。它允許開發者透過建構符號圖 (Symbolic Graph) 來表示複雜的神經網路結構或線性代數運算，並由底層引擎自動選擇最佳的硬體加速路徑。若開發者在 PyTorch 等框架中採用即時執行 (Eager Mode) 呼叫 MPS 後端，每一個運算子（如矩陣乘法、啟動函數）都會獨立觸發一次指令緩衝區的建立與排程，這在微秒級的運算中會導致驚人的開銷累積 。MPSGraph 的核心價值在於其靜態編譯與最佳化能力，能將整張計算圖編譯為一個 MPSGraphExecutable。編譯器最佳化通道與描述符在編譯階段，MPSGraph 引擎會透過 MPSGraphCompilationDescriptor 控制最佳化層級 。系統會執行多個最佳化通道 (Optimization Passes)：運算子融合 (Operator Fusion)： 這是降低啟動開銷最有效的方法。引擎會自動識別相鄰的逐元素運算 (Pointwise Operations)、線性轉換與正規化層，將它們合併為單一的 GPU 運算核心，徹底消除中間暫存張量的記憶體讀寫與核心啟動間隙 。常數摺疊 (Constant Folding)： 對於輸入皆為常數的子圖，編譯器會在編譯期直接計算出結果，避免在執行期耗費運算資源 。靜態記憶體規劃： 透過分析圖中張量的生命週期，預先配置記憶體，避免執行期的動態記憶體分配延遲。開發者可以透過設定 MPSGraphOptions 與 MPSGraphOptimizationProfile 來平衡編譯時間與執行效能。例如，在生產環境中，開發者可以設定更激進的優化層級，以較長的初始編譯時間換取更低的推論延遲 。此外，透過啟用 reducedPrecisionFastMath，可以在不影響模型收斂精度的前提下，允許編譯器採用速度更快的近似數學指令 。精度轉換陷阱與同步開銷在使用 MPSGraph 時，一個違反直覺的現象是浮點精度的選擇。在傳統的 NVIDIA 架構中，將模型轉換為 float16 (半精度) 幾乎總是能提升效能。然而在 Apple Silicon 上，若未正確對齊硬體管線，將 float32 模型強制轉換為 float16 反而可能導致推論時間延長（例如從 18 秒增加至 22 秒）。這是因為架構內部若缺乏對應的最佳化路徑，軟體層級的精度轉換開銷將超過其帶來的傳輸頻寬增益 。另一個嚴重的效能陷阱是過度同步。根據 Apple 官方的效能調校指南，當主機程式將工作提交給 MPS 佇列後，絕對不應該立即呼叫 waitUntilCompleted() 來等待結果 。僅僅是等待一個空的指令緩衝區完成，就可能產生長達 2.5 毫秒的延遲。正確的實踐是：在等待前一個緩衝區完成的同時，CPU 應該已經開始編碼並提交下一個緩衝區。這種異步並行的設計模式，能夠將整體系統吞吐量提升高達 10 倍 。MLX 框架對 Metal 管線的封裝與排程機制Apple 推出的開源陣列框架 MLX，是專為 Apple Silicon 機器學習研究所設計的里程碑式專案 。與依賴 MPS 作為翻譯層的 PyTorch 不同，MLX 直接針對 Apple 的 UMA 架構與 Metal API 進行了深度封裝與客製化 。透過分析 MLX 的官方文件與 GitHub 儲存庫（包含 Pull Requests 與 Issues），我們可以清晰地看出其架構設計如何系統性地解決 GPU 核心啟動開銷問題。延遲評估 (Lazy Evaluation) 機制的實作MLX 效能優勢的基石在於其「延遲評估」(Lazy Evaluation) 架構。在傳統的 Python 科學運算中（如 NumPy 或 PyTorch Eager Mode），當開發者撰寫一行加法程式碼時，系統會立即分配記憶體並呼叫底層函式進行計算。而在 MLX 中，呼叫數學運算函式並不會觸發任何實質的 GPU 運算，而是會在背景記憶體中動態建立一個有向無環圖 (Directed Acyclic Graph, DAG)，記錄下運算節點與資料依賴關係 。這張計算圖會持續生長，直到滿足特定條件時才會觸發「具現化」(Materialization)。這些觸發條件包含：明確呼叫 mx.eval()、嘗試列印張量內容至終端機、將張量轉換為 NumPy 陣列、或將張量儲存至磁碟 。延遲評估機制在降低排程開銷上帶來了兩大優勢：批次處理 API 邊界跨越： 透過推遲執行，MLX 將數百次從 Python 跨越至 C++ 再跨越至 Metal 驅動程式的 API 呼叫，壓縮為單一的批次提交動作，從根本上削弱了直譯器 (Interpreter) 造成的延遲 。精確的記憶體生命週期管理： 由於 MLX 在執行前已經掌握了完整的計算圖，它可以精確計算出中間暫存變數 (Intermediate Variables) 的生成與消亡時間。這使得 MLX 的記憶體配置器能夠在統一記憶體池中重複利用記憶體區塊，避免在關鍵執行路徑上進行昂貴的系統層級動態記憶體分配 (Dynamic Memory Allocation) 。獨立的背景排程器與執行緒隔離在多程序或非同步程式設計環境中，主機端 CPU 的阻塞 (Blocking) 往往會導致 GPU 處於飢餓狀態。為了確保 GPU 佇列的持續飽和，MLX 在底層 C++ 實作中摒棄了依賴 Python 全局直譯器鎖 (GIL) 的排程方式。當 mx.eval() 被觸發時，MLX 會將計算圖的評估任務派發給一個專屬的背景工作執行緒 (Worker Thread) 或客製化的 C++ 執行緒池 。這種架構設計保證了 Metal 指令緩衝區的建立與編碼過程完全獨立於主應用程式。因此，即使開發者在 Python 端處理網路請求或檔案 I/O，背景的 MLX 排程器仍能持續將指令餵給 GPU 硬體，實現真正的 CPU-GPU 非同步並行 。MLX 框架特性內部實作機制降低開銷的具體作用延遲評估 (Lazy Evaluation)動態構建 DAG，僅在 mx.eval() 時執行 。消除中間 Python-to-C++ 的 API 呼叫成本與動態記憶體分配 。背景非同步排程器專屬 C++ 執行緒池，繞過 Python GIL 。確保 GPU 持續獲得指令餵給，避免因主執行緒阻塞導致 GPU 閒置 。統一記憶體 (UMA) 感知零複製 (Zero-Copy) 張量運算 。避免傳統 PCIe 匯流排環境下昂貴的 Host-to-Device 資料傳輸 。MLX 的 JIT 編譯與自訂 Metal 核心整合實務僅靠延遲評估將多個運算打包提交，仍然意味著計算圖中的每一個節點都會對應到一個獨立的 Metal 核心啟動。為了進一步壓榨效能，MLX 提供了即時編譯 (JIT Compilation) 機制與自訂核心 API，這是縮減微觀層面開銷的最終手段。巨型核心融合：mx.compile 的機制與挑戰開發者可以透過 @mx.compile 裝飾器 (Decorator) 標譯純函數。當該函數首次被呼叫時，MLX 不會直接執行其中的運算，而是會遍歷其產生的計算圖，並執行強制的運算子融合 (Operator Fusion)。它會動態生成並編譯一個高度最佳化的 Metal Shading Language 著色器，將原本需要多次啟動的迴圈、正規化層 (Normalization) 或注意力機制 (Attention)，整合成單一個巨型核心 (Megakernel) 。這項技術的威力在效能測試中展露無遺。在針對微型 Transformer 模型 (microGPT) 的基準測試中，單純依賴基礎 MLX GPU 運算的實作，由於密集的核心啟動開銷，其吞吐量甚至低於 C 語言撰寫的純 CPU 實作 。只有透過嚴格的 JIT 融合消除這些開銷，GPU 的硬體優勢才能顯現。shapeless=True 的應用與邊角案例預設情況下，mx.compile 會對輸入張量的形狀 (Shape) 與資料型態 (Data Type) 進行嚴格的靜態綁定。如果函數在後續呼叫中接收到不同維度的輸入，MLX 會強制觸發極為耗時的重新編譯 (Recompilation) 。這在處理大型語言模型推論時是致命的，因為每次生成新詞元時，KV Cache 的序列長度都會增加。為了適應動態形狀，開發者必須傳遞 shapeless=True 參數給編譯器，指示 MLX 產生接受可變維度的泛用型 MSL 程式碼 。然而，這項功能在極端邊角案例中曾暴露出框架的底層缺陷。在 MLX 的 GitHub Issue #3201 中，開發者發現當在 shapeless=True 的編譯函數中，對動態長度的輸入使用歸約運算 (Reduction，如 sum 或 mean) 時，函數會回傳第一次呼叫時的過期快取結果 。除錯過程揭露了其核心原因：歸約核心的輸出形狀通常是固定的純量 (Scalar, ``)。在重新評估計算圖時，編譯器因為發現輸出形狀未改變，錯誤地決定不再重新派發 (Re-dispatch) 歸約核心，而是直接使用了依據第一次呼叫長度所建立的快取中介緩衝區 。這個案例凸顯了在動態巨型核心中管理記憶體邊界的複雜性。整合 mx.fast.metal_kernel 進行底層控制對於無法透過高階函數有效融合的特殊演算法，MLX 提供了 mx.fast.metal_kernel API，允許開發者直接將自訂的 C++ MSL 字串注入運算管線中 。這項功能自動處理了繁瑣的引數緩衝區建立與函式簽章 (Function Signature) 生成。當開發者傳遞名為 inp 的 float16 陣列時，MLX 會在背景自動將其轉換為 MSL 的 const device float16_t* inp，並自動附加 inp_shape 等輔助變數 。為了在自訂核心中實現反向傳播 (Backward Pass) 與梯度累積，開發者可以設定 atomic_outputs=True。這項設定會自動將輸出陣列轉換為 device atomic_float\* 型態，允許自訂核心使用 Metal 原生的硬體原子加法 (Atomic Add) 功能。同時，配合 init_value=0 參數，MLX 會在核心啟動前自動清空輸出緩衝區，確保平行梯度累積的正確性 。在自訂核心的排程上，開發者必須親自指定 grid (執行網格) 與 threadgroup (執行緒群組) 的尺寸 。根據 Apple Silicon 的架構特性，最佳實踐是將 threadgroup 的維度設定為 32 的倍數（對齊 SIMD-group 的寬度），以避免硬體層級的執行緒發散 (Warp Divergence) 與運算資源浪費 。針對 M2 Pro 等 Apple Silicon 晶片的具體策略與程式碼架構建議綜合上述學術研究、Apple 官方文件以及 MLX 框架的工程實踐，針對在 M2 Pro 或同等等級的 Apple Silicon 環境下開發高效能機器學習或運算密集型應用，本文總結出以下降低 Kernel Launch Overhead 的具體策略與程式碼架構建議：1. 貫徹統一記憶體架構 (UMA) 的零複製原則避免任何邏輯上的 Host-to-Device 傳輸。在使用原生 Metal API 時，應直接在 MTLBuffer 或 MTLTexture 上建立資料，並利用指標共享機制。若整合 MPSGraph，在將 Metal 資源轉換為計算圖張量時，應使用封裝現有緩衝區指標的 MPSGraphTensorData 建構子，嚴格避免額外的記憶體配置與拷貝操作 。2. 利用 MLX 的巨型核心融合與動態編譯在建構深度神經網路推論伺服器（如 SGLang 的 MLX 後端整合）時，必須徹底放棄逐層呼叫運算子的作法 。模型前向傳遞封裝： 確保整個模型的前向傳遞邏輯（或至少是單一的 Transformer Block）被封裝在標記有 @mx.compile 的函數中 。動態形狀處理： 針對推論過程中長度不斷變化的 KV Cache，在編譯時務必加上 shapeless=True，以避免每生成一個詞元就引發長達數毫秒的重新編譯停頓 。混合精度最佳化： 將模型權重與 KV Cache 量化為 8-bit 或 4-bit 格式，不僅能大幅降低記憶體佔用，更能減少資料移動造成的頻寬瓶頸，這在依賴記憶體讀取速度的 LLM 解碼階段 (Decode Phase) 尤為關鍵 。3. 建構低開銷的 Metal 原生指令管線若專案需求超越了 MLX 的能力範圍，必須退回使用純 Swift 與 Metal API，則程式碼架構應遵循以下嚴格的低開銷守則：引數緩衝區扁平化： 使用 MTLArgumentEncoder 將所有張量指標、常數與紋理預先打包為單一個資源陣列。確保運算迴圈內的 setBuffer 呼叫次數為 $O(1)$ 。解耦的指令佇列： 採用 Metal 4 的 MTL4CommandQueue，並透過多執行緒背景佇列預先呼叫 makeCommandBuffer() 生成指令。在關鍵迴圈中，利用群組提交功能，降低作業系統層級的執行緒切換與排程延遲 。異步事件管理： 嚴禁使用 waitUntilCompleted() 來阻塞 CPU。應實作多重緩衝 (Multi-buffering) 架構，讓 CPU 永遠領先 GPU 進行下一個批次的指令編碼 。4. 採用間接指令緩衝區 (ICB) 實現 GPU 驅動排程對於需要頻繁依據條件動態調整運算配置的演算法，徹底拔除 CPU 介入是唯一的解法 。GPU 內部編碼： 設計一個專門的主控運算核心 (Master Compute Kernel)，由它在硬體內部解析動態參數，並將 concurrentDispatchThreads 指令直接寫入透過 MTLIndirectCommandBufferDescriptor 分配的 ICB 中 。執行前最佳化： 若 ICB 中的指令模式固定但包含冗餘（例如條件篩選後產生的大量空白指令），在呼叫 executeCommandsInBuffer 前，應發送 optimizeIndirectCommandBuffer 請求。這將指示硬體緊湊化排程狀態，儘管這會限制指令範圍的靈活性，但能顯著提升最終的執行吞吐量 。5. 運用 MSL 原生矩陣與原子指令替代組合語言開發者應停止依賴任何未公開的 Apple IR 或 \_\_asm 內聯語法，以確保相容性並獲得 LLVM 編譯器的最佳化紅利 。硬體 MMA 加速： 將關鍵的張量收縮與矩陣乘法重寫為依賴 metal::simdgroup_matrix 的實作，確保運算在 32 KiB 執行緒群組記憶體內高效流轉，完全避免暫存器溢出 (Register Spill) 。無鎖平行歸約： 利用 MSL 的 atomic_fetch_add_explicit 與 atomic_ulong（若有 64 位元需求）實作跨群組的記憶體寫入，這能避免在複雜的自訂計算圖節點中引入昂貴的記憶體屏障，實現高吞吐量的資料聚合 。透過整合上述架構，從高階的 MLX 延遲評估與 JIT 融合，到中階的 ICB GPU 驅動排程，再到底層的 UMA 記憶體管理與 MSL 原生指令最佳化，開發者將能夠在 Apple Silicon 平台上徹底擊破 Kernel Launch Overhead 的瓶頸，充分釋放 M2 Pro 等硬體的頂級運算潛能。

# Apple M2 Pro 小模型推論最佳化 API 對照表

## Metal / MLX / MPSGraph / Swift / Apple GPU 最佳化速查

---

# 1. 核心瓶頸（M2 Pro 小模型）

| 問題                   | 原因                      | 現象                 |
| ---------------------- | ------------------------- | -------------------- |
| Kernel Launch Overhead | dispatch 太多             | GPU utilization 很低 |
| Command Encoding Cost  | CPU encode command buffer | decode tok/s 卡住    |
| GPU↔CPU Sync           | `eval()` / `.item()`      | token latency 暴增   |
| Memory-bound           | 小 matmul 太碎            | GPU 算力吃不滿       |
| Intermediate Buffers   | tensor 過多               | unified memory 壓力  |

---

# 2. 最有效策略（重要度排序）

| 優先級     | 策略                  | 效果                 |
| ---------- | --------------------- | -------------------- |
| ⭐⭐⭐⭐⭐ | Fused Kernels         | 大幅減少 dispatch    |
| ⭐⭐⭐⭐⭐ | Lazy Evaluation       | 減少 materialization |
| ⭐⭐⭐⭐   | 預編譯 Pipeline       | 降低 runtime compile |
| ⭐⭐⭐⭐   | 避免 GPU↔CPU sync     | 降低 token latency   |
| ⭐⭐⭐⭐   | ICB / Argument Buffer | 降低 CPU encode      |
| ⭐⭐⭐     | FP16 / SIMD-group     | 提高 occupancy       |
| ⭐⭐⭐     | MPSGraphExecutable    | 固定 graph compile   |
| ⭐⭐       | Triple Buffering      | 隱藏 command latency |

---

# 3. Metal API 最佳化

## (A) Pipeline 預編譯

### ❌ 不好

```swift
let pipeline = try device.makeComputePipelineState(function: fn)
```

decode loop 每次建立。

---

### ✅ 正確

```swift
// App startup
pipelineCache["rmsnorm"] =
    try device.makeComputePipelineState(function: fn)
```

## 效果

- 避免 runtime compile
- 降低 stutter
- 減少 first-token latency

---

# 4. Command Buffer 最佳化

## ❌ 不好

```swift
for layer in layers {
    let cb = queue.makeCommandBuffer()!
    ...
    cb.commit()
}
```

問題：

- command buffer 過碎
- CPU encode cost 高

---

## ✅ 正確

```swift
let cb = queue.makeCommandBuffer()!

for layer in layers {
    encodeLayer(...)
}

cb.commit()
```

## 建議

- 一個 token → 一個 command buffer
- 不要 layer 一個 command buffer

---

# 5. Argument Buffer（超重要）

## 問題

大量：

```swift
setBuffer(...)
setBuffer(...)
setBuffer(...)
```

CPU call 太多。

---

## ✅ 改成 Argument Buffer

```metal
struct Args {
    device half* q;
    device half* k;
    device half* v;
};
```

```swift
encoder.setBuffer(argBuffer, offset: 0, index: 0)
```

## 效果

- 降低 CPU encode
- 減少 bind overhead
- decode latency 明顯下降

---

# 6. Indirect Command Buffer (ICB)

## 適合

- autoregressive decode
- 重複 transformer layers
- 固定 dispatch pattern

---

## API

```swift
let icb = device.makeIndirectCommandBuffer(
    descriptor: desc,
    maxCommandCount: N,
    options: []
)
```

---

## 優勢

| 一般 dispatch      | ICB         |
| ------------------ | ----------- |
| CPU 每次 encode    | encode once |
| CPU scheduling     | GPU-driven  |
| 高 launch overhead | 低 overhead |

---

# 7. MLX 最重要 API

---

## (A) Lazy Evaluation

### ❌ 不好

```python
x = layer(x)
mx.eval(x)
```

每層 sync。

---

### ✅ 正確

```python
for layer in layers:
    x = layer(x)

mx.eval(x)
```

---

## (B) `mx.compile`

### ❌

dynamic eager graph

---

### ✅

```python
@mx.compile
def forward(x):
    ...
```

## 效果

- kernel fusion
- graph optimization
- fewer dispatches

---

# 8. MLX Swift 注意事項

## ❌ 高成本 API

```swift
array.item(Int.self)
```

會：

- 強制 GPU sync
- 阻塞 decode

---

## ✅ 正確

盡量：

- 延後讀回
- batch decode
- 避免 token-by-token CPU sync

---

# 9. MPSGraph 最佳化

---

## MPSGraphExecutable

### ✅ 預編譯 graph

```swift
let exec = try graph.compile(
    with: device,
    feeds: feeds,
    targetTensors: outputs,
    targetOperations: nil
)
```

---

## 推論

```swift
exec.runAsync(...)
```

---

## 效果

| 一般 MPSGraph   | Executable  |
| --------------- | ----------- |
| runtime compile | precompiled |
| graph rebuild   | reusable    |
| 高 CPU 開銷     | 低開銷      |

---

# 10. Fused Kernel（最重要）

---

## ❌ 傳統

```text
RMSNorm
↓
Linear
↓
Residual Add
↓
Activation
```

= 4 dispatches

---

## ✅ Fused

```text
[FUSED RMSNorm + Linear + Residual]
```

= 1 dispatch

---

## 最值得融合

| Fused Block         | 效果       |
| ------------------- | ---------- |
| RMSNorm + Linear    | ⭐⭐⭐⭐⭐ |
| Dequant + Matmul    | ⭐⭐⭐⭐⭐ |
| QKV Projection      | ⭐⭐⭐⭐   |
| Softmax + Attention | ⭐⭐⭐⭐   |
| Residual + Norm     | ⭐⭐⭐⭐   |

---

# 11. Apple GPU Shader 最佳化

## FP16

```metal
half
half4
half8
```

優先於：

```metal
float
```

---

## SIMD-group

```metal
simdgroup_barrier(...)
```

適合：

- reduction
- attention
- RMSNorm

---

## Threadgroup Memory

```metal
threadgroup half shared[256];
```

用途：

- tile matmul
- reduction cache

---

# 12. 小模型最佳 Thread 配置（M2 Pro）

| Kernel        | 建議 threadgroup   |
| ------------- | ------------------ |
| RMSNorm       | 128                |
| Attention     | 256                |
| Matmul 小矩陣 | 128~256            |
| Reduction     | SIMD-group aligned |

---

# 13. Decode 最佳架構（推薦）

```text
CPU
 └─ Token Scheduler
      ↓
Command Buffer
 ├─ Fused RMSNorm+Linear
 ├─ Fused Attention
 ├─ Fused MLP
 └─ Logits

GPU
 └─ Single eval()

CPU
 └─ Sample next token
```

---

# 14. 目前 MLX 真實瓶頸（社群 issue）

| 問題               | 影響          |
| ------------------ | ------------- |
| 每 token 多次 sync | latency 爆炸  |
| evalLock           | serialization |
| dispatch 太碎      | GPU idle      |
| unfused attention  | memory-bound  |

---

# 15. M2 Pro 最終最佳化策略（實戰）

## 第一階段（最大收益）

✅ Fused kernels
✅ Lazy eval
✅ 避免 `.item()`
✅ 減少 dispatch

---

## 第二階段

✅ Argument buffers
✅ ICB
✅ Triple buffering
✅ MPSGraphExecutable

---

## 第三階段（極限）

✅ Custom Metal kernels
✅ Tile matmul
✅ SIMD-group reduction
✅ Offline shader compile

---

# 16. 最終核心結論

## Apple Silicon 小模型推論真正瓶頸不是 FLOPS

而是：

```text
CPU encode
+
kernel launch
+
GPU scheduling
+
sync overhead
```

---

# 17. 最有效的一句話

## 「把 100 個小 dispatch 變成 5 個大 dispatch」

這通常比：

```text
單純再優化 matmul FLOPS
```

更有效。
