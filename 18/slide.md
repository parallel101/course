---
marp: true
theme: cache-conflict
size: 16:9
paginate: true
footer: Parallel101 · Cache Set Conflict
---

<!-- _class: lead -->
<!-- _paginate: false -->
<!-- _footer: "" -->

<div class="eyebrow">MEMORY / CACHE / STRIDE</div>

# 越对齐，越慢？

<div class="lead-mark">
  <span>nj</span>
  <b>1024</b>
  <i>3.09×</i>
</div>

<!--
口播提示：先抛反直觉结果。不要解释 cache geometry，问大家为什么 1024 反而最慢。
-->

---

# 我们实际在怎么访存？

<div class="visit-layout">
<div class="code-card">

```cpp
for (int j = 0; j < nj; ++j)
    for (int i = 0; i < ni; ++i)
        sum += matrix[i * nj + j];
```

<div class="stride-formula">stride = nj × 4 B</div>
</div>

<div class="matrix-figure">
  <div class="matrix-label rows">i</div>
  <div class="matrix-label cols">j</div>
  <div class="matrix-cells">
    <div class="scan-line">
      <i></i><i></i><i></i><i></i><i></i><i></i>
      <div class="scan-arrow">↓</div>
    </div>
  </div>
</div>
</div>

<!--
口播提示：矩阵按行连续存储，但内层 i 在变，所以是在各行同一列之间跳。求和本身不重要，这是垂直遍历的模型。
-->

---

# 看起来应该越整齐越快

<div class="alignment-stair">
  <div class="align-step s1"><span>512</span><i>2 KiB</i></div>
  <div class="align-step s2"><span>1024</span><i>4 KiB</i></div>
  <div class="align-step s3"><span>2048</span><i>8 KiB</i></div>
  <div class="align-step s4"><span>4096</span><i>16 KiB</i></div>
</div>

<div class="intuition-arrow"><span>2 的幂</span><b>→</b><span>“CPU 喜欢对齐”</span></div>

<!--
口播提示：这是常见直觉，不要急着批判。这里的数字是每行 int 数量，下面是对应字节步长。
-->

---

# 实测：1024 是一根针

<div class="chart-shell">
<svg class="sweep-chart" viewBox="0 0 1000 420" role="img" aria-label="nj 从 1008 到 1040 的延迟曲线，1024 处出现尖峰">
  <line x1="70" y1="350" x2="920" y2="350" class="axis"/>
  <line x1="70" y1="60" x2="70" y2="350" class="axis"/>
  <line x1="70" y1="339.6" x2="920" y2="339.6" class="grid"/>
  <line x1="70" y1="236.1" x2="920" y2="236.1" class="grid"/>
  <line x1="70" y1="132.5" x2="920" y2="132.5" class="grid"/>
  <text x="52" y="345" text-anchor="end">50</text>
  <text x="52" y="241" text-anchor="end">100</text>
  <text x="52" y="137" text-anchor="end">150</text>
  <text x="52" y="72" text-anchor="end">μs</text>
  <text x="70" y="382" text-anchor="middle">1008</text>
  <text x="495" y="382" text-anchor="middle" class="hot-text">1024</text>
  <text x="920" y="382" text-anchor="middle">1040</text>
  <polyline class="series" points="70.0,340.5 96.6,328.3 123.1,335.1 149.7,329.3 176.2,339.9 202.8,329.0 229.4,335.0 255.9,328.8 282.5,341.9 309.1,328.2 335.6,334.4 362.2,327.4 388.8,338.7 415.3,327.2 441.9,332.5 468.4,309.7 495.0,76.5 521.6,307.2 548.1,331.5 574.7,325.9 601.2,336.0 627.8,314.6 654.4,324.2 680.9,324.0 707.5,341.3 734.1,326.5 760.6,333.0 787.2,314.7 813.8,327.2 840.3,317.7 866.9,322.9 893.4,324.2 920.0,336.6"/>
  <line x1="495" y1="60" x2="495" y2="350" class="hot-line"/>
  <circle cx="495" cy="76.5" r="10" class="hot-dot"/>
</svg>
<div class="spike-card"><span>nj = 1024</span><b>177.047 μs</b><small>周边约 49–66 μs</small></div>
</div>

<div class="evidence-strip"><span>ni = 128</span><span>31 samples · median</span><span>GCC 16.1 · -O3 -march=native</span><span>i7-9750H</span></div>

<!--
口播提示：这张图只是发现实验。nj 改变时逻辑矩阵大小和物理 stride 一起变了，所以它还不是因果证明。
-->

---

# 为什么 Cache 值得存在？

<div class="reuse-curve">
  <div class="reuse-axis y">访问频率</div>
  <div class="reuse-axis x">变量</div>
  <div class="reuse-bar b1"><b>200</b><span>a</span></div>
  <div class="reuse-bar b2"><b>100</b><span>b</span></div>
  <div class="reuse-bar b3"><b>50</b><span>c</span></div>
  <div class="reuse-bar b4"><b>25</b><span>d</span></div>
  <div class="reuse-bar b5"><b>12</b><span>e</span></div>
  <div class="reuse-tail">······</div>
</div>

<div class="hot-cold"><span>少量热数据</span><i></i><span>大量冷数据</span></div>

<!--
口播提示：这只是局域性的示意图，不是这段 benchmark 的测量。可讲热点、复用与层级结构的故事。
-->

---

# 一次读取，逐级查找

<div class="lookup-path">
  <div class="lookup-node l1"><small>1</small><b>L1D</b><span>32 KiB / core</span></div>
  <div class="lookup-link"><b>MISS</b><i>→</i></div>
  <div class="lookup-node l2"><small>2</small><b>L2</b><span>256 KiB / core</span></div>
  <div class="lookup-link"><b>MISS</b><i>→</i></div>
  <div class="lookup-node l3"><small>3</small><b>L3</b><span>12 MiB shared</span></div>
  <div class="lookup-link"><b>MISS</b><i>→</i></div>
  <div class="lookup-node ram"><small>4</small><b>DRAM</b><span>far away</span></div>
</div>

<div class="return-path">cache line <b>←</b> cache line <b>←</b> cache line <b>←</b></div>

<!--
口播提示：从 L1 开始，miss 才继续向下。主存取回的是整条 cache line，并向上填充。
-->

---

# 这台 i7-9750H 的 Cache 拓扑

<div class="cpu-topology">
  <div class="cores">
    <div><b>Core 0</b><span>L1D 32 KiB</span><span>L2 256 KiB</span></div>
    <div><b>Core 1</b><span>L1D 32 KiB</span><span>L2 256 KiB</span></div>
    <div><b>Core 2</b><span>L1D 32 KiB</span><span>L2 256 KiB</span></div>
    <div><b>Core 3</b><span>L1D 32 KiB</span><span>L2 256 KiB</span></div>
    <div><b>Core 4</b><span>L1D 32 KiB</span><span>L2 256 KiB</span></div>
    <div><b>Core 5</b><span>L1D 32 KiB</span><span>L2 256 KiB</span></div>
  </div>
  <div class="shared-cache"><b>L3 · 12 MiB</b><span>6 cores shared</span></div>
  <div class="memory-bus"></div>
  <div class="dram-box">DRAM</div>
</div>

<!--
口播提示：一二级每个物理核各一份，三级由 6 个核共享。12 个逻辑 CPU 来自 SMT，不是 12 份 L1/L2。
-->

---

# 软件里的“理想 Cache”

<div class="ideal-cache-layout">
<div class="ideal-cache-code">

```cpp
unordered_map<Address, CacheLine> cache;

read(address):
    if (cache.contains(address))
        return cache[address];

    line = memory.read_line(address);
    cache[address] = line;
    return line;
```

</div>
<div class="ideal-map">
  <div class="ideal-key">0x7f…c040</div><i>→</i><div class="ideal-value">64 B line</div>
  <div class="ideal-key">0x7f…d040</div><i>→</i><div class="ideal-value">64 B line</div>
  <div class="ideal-key">0x7f…e040</div><i>→</i><div class="ideal-value">64 B line</div>
  <div class="ideal-more">⋮</div>
</div>
</div>

<div class="hardware-budget"><span>面积</span><b>×</b><span>功耗</span><b>×</b><span>延迟</span></div>

<!--
口播提示：先把 cache 想成地址到 cache line 的字典，这个模型最容易理解。但硬件不能照搬软件 unordered_map：查 L1 的结构本身必须极小、极快。
-->

---

# 查 abandon，不必翻完整本字典

<div class="dictionary-lookup">
  <div class="unsorted-book">
    <div class="book-title">512 words</div>
    <div class="word-noise">zoo · kernel · cache · apple · matrix · way · abandon · line · address · set · memory · core · …</div>
    <div class="linear-cost">逐个比较</div>
  </div>
  <div class="lookup-transform">→</div>
  <div class="directory-book">
    <div class="directory-list">
      <div class="letter active">A</div><div class="page-range active">1–20</div>
      <div class="letter">B</div><div class="page-range">21–35</div>
      <div class="letter">C</div><div class="page-range">36–52</div>
      <div class="letter muted">⋮</div><div class="page-range muted">⋮</div>
      <div class="letter">Z</div><div class="page-range">498–512</div>
    </div>
    <div class="a-page"><b>A</b><span>abandon</span><span>address</span><span>apple</span></div>
  </div>
</div>

<div class="dictionary-hash">首字母 <b>A</b> → 只查 A 的范围</div>

<!--
口播提示：用字典目录建立直觉。目录不需要理解复杂哈希；它只是从单词里取一个很便宜的特征，把搜索范围缩小。
-->

---

# Set 是桶，Way 是桶里的槽位

<div class="set-way-analogy">
  <div class="word-hash">
    <span>abandon</span>
    <i>取首字母</i>
    <b>A</b>
  </div>
  <div class="analogy-arrow">→</div>
  <div class="one-set">
    <div class="set-name">set A</div>
    <div class="way-labels"><span>way 0</span><span>way 1</span><span>way 2</span><span>way 3</span><span>way 4</span><span>way 5</span><span>way 6</span><span>way 7</span></div>
    <div class="word-ways"><i>abandon</i><i>address</i><i>apple</i><i>array</i><i>align</i><i>alias</i><i>atom</i><i>AVX</i></div>
    <div class="parallel-compare"><b>↑</b><b>↑</b><b>↑</b><b>↑</b><b>↑</b><b>↑</b><b>↑</b><b>↑</b><span>parallel tag compare</span></div>
  </div>
</div>

<!--
口播提示：set 对应目录桶，way 对应桶内固定槽位。这里不要说成真实链表：硬件会并行比较这个 set 里的多个 tag，way 数限制的是候选槽位数量。
-->

---

# 512 个位置，为什么只装下 8 个词？

<div class="capacity-contrast">
  <div class="balanced-capacity">
    <div class="capacity-title">均匀分布</div>
    <div class="mini-set-grid">
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
    </div>
    <div class="capacity-total"><b>64 sets × 8 ways</b><span>512 lines</span></div>
  </div>
  <div class="all-a-capacity">
    <div class="capacity-title">全部以 A 开头</div>
    <div class="a-stream">A A A A A A A A A A A A A A A A …</div>
    <div class="a-bucket"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
    <div class="capacity-miss"><b>8</b><span>留下</span><em>+504 MISS</em></div>
  </div>
</div>

<!--
口播提示：总位置数没有变，失败来自分布。所有 key 都落进一个 set 时，其他 63 个 set 再空也借不过来；这就是 associativity conflict 的核心。
-->

---

# 把“字典”换成本机 L1D

<div class="line-derivation">
  <div class="derive-step"><b>32 KiB</b><span>L1D capacity</span></div>
  <i>÷</i>
  <div class="derive-step"><b>64 B</b><span>cache line</span></div>
  <i>=</i>
  <div class="derive-step cyan-step"><b>512</b><span>lines</span></div>
</div>

<div class="split-arrow">↓</div>

<div class="set-way-product">
  <div><b>64</b><span>sets</span></div>
  <i>×</i>
  <div><b>8</b><span>ways / set</span></div>
  <i>=</i>
  <div><b>512</b><span>lines</span></div>
</div>

<!--
口播提示：这些数字来自当前 i7-9750H 的 sysfs：L1D 32 KiB、64-byte line、64 sets、8-way。先算 512 条 line，再解释它们为什么必须排成 64×8。
-->

---

# 目录编号，直接从地址里取

<div class="hardware-index">
  <div class="bit-source">
    <div class="address-bits">
      <div class="tag-bits">tag</div>
      <div class="set-bits">set [11:6]</div>
      <div class="offset-bits">offset [5:0]</div>
    </div>
    <div class="bit-pick">取 6 bits <b>↓</b></div>
    <div class="map-formula">set(address) = (address / 64) mod 64</div>
  </div>
  <div class="index-arrow">→</div>
  <div class="indexed-set">
    <div class="indexed-title">selected set</div>
    <div class="indexed-ways"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
    <div class="indexed-caption">8 tags compared together</div>
  </div>
</div>

<!--
口播提示：L1 不做软件式哈希计算，直接用地址低位中的 6 bits 选 64 个 sets 之一，再并行比较 8 个 tags。这里也为下一节 VIPT 埋伏笔：offset 与 L1 set bits 都落在 4 KiB page offset 内。
-->

---

# Set 缩小范围，Tag 才负责认人

<div class="tag-match-layout">
  <div class="query-address">
    <div class="query-label">load address</div>
    <div class="query-bits"><span>tag = 0x19a</span><b>set = 17</b><i>offset</i></div>
  </div>
  <div class="tag-flow">↓</div>
  <div class="tag-way-table">
    <div class="tag-table-head"><span>way</span><span>stored tag</span><span>64 B line</span><span>match</span></div>
    <div><span>0</span><code>0x02f</code><i></i><em>×</em></div>
    <div><span>1</span><code>0x19a</code><i></i><em class="match">✓</em></div>
    <div><span>2</span><code>0x8c1</code><i></i><em>×</em></div>
    <div><span>…</span><code>…</code><i></i><em>×</em></div>
    <div><span>7</span><code>0xf41</code><i></i><em>×</em></div>
  </div>
</div>

<div class="tag-storage-note">set 已知 · line 对齐已知 · 每个 way 只需保存 <b>tag + line</b></div>

<!--
口播提示：访问地址的 offset 不一定为 0；隐含的是每条缓存行的基地址按 64 B 对齐。set 已经由索引决定、offset 只选 line 内字节，所以 way 元数据保存高位 tag，用物理 tag 做最终消歧义。
-->

---

# 4096 B：每一行都撞进同一个 Set

<div class="conflict-layout">
  <div class="stride-math bad-math">
    <span>4096 B</span>
    <i>÷ 64 B</i>
    <b>64 lines</b>
    <em>64 mod 64 = <strong>0</strong></em>
  </div>
  <div class="row-sequence">
    <span>row 0</span><i>→</i><b>set 0</b>
    <span>row 1</span><i>→</i><b>set 0</b>
    <span>row 2</span><i>→</i><b>set 0</b>
    <span>⋮</span><i> </i><b>⋮</b>
    <span>row 127</span><i>→</i><b>set 0</b>
  </div>
  <div class="set-overflow">
    <div class="ways-title">set 0 · 8 ways</div>
    <div class="way-slots"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
    <div class="overflow-lines">+120</div>
  </div>
</div>

<!--
口播提示：内层一次扫过 128 行，128 条 line 争抢 set 0 的 8 个 way。总容量看起来够也没用。
-->

---

# 4160 B：每一行轮转到下一个 Set

<div class="spread-layout">
  <div class="stride-math good-math">
    <span>4160 B</span>
    <i>÷ 64 B</i>
    <b>65 lines</b>
    <em>65 mod 64 = <strong>1</strong></em>
  </div>
  <div class="set-carpet">
    <div class="set-route">0 <i>→</i> 1 <i>→</i> 2 <i>→</i> 3 <i>→</i> ··· <i>→</i> 63</div>
    <div class="set-grid">
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
    </div>
    <div class="set-grid-labels"><span>set 0</span><span>64 sets</span><span>set 63</span></div>
  </div>
</div>

<!--
口播提示：加 16 个 int，也就是多一个 cache line 的 row stride。现在相邻行依次经过 64 个 sets，两轮正好覆盖 128 行。
-->

---

# 一条 Cache Line 里有 16 个 int

<div class="line-reuse">
  <div class="line-pass"><b>j = 0</b><div class="int-line first"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div></div>
  <div class="reuse-down"><span>下一列还要用同一条 line</span><b>↓</b></div>
  <div class="line-pass"><b>j = 1</b><div class="int-line second"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div></div>
</div>

<div class="eviction-compare">
  <div class="evicted"><span>4096 B</span><b>evicted</b><i>×</i></div>
  <div class="survived"><span>4160 B</span><b>still cached</b><i>✓</i></div>
</div>

<!--
口播提示：问题不是一次读第 0 列，而是读第 1 列时，本该复用刚取回的 line。冲突会在复用前把它挤走。
-->

---

# 控制变量：逻辑矩阵完全不变

<div class="fixed-badge">128 × 1024 · same values · same reads</div>

<div class="result-bars">
  <div class="result-row bad-result">
    <div class="result-label"><b>4096 B</b><span>+0 int</span></div>
    <div class="result-track"><i style="width: 100%"></i></div>
    <div class="result-value"><b>175.271</b><span>μs</span><small>[173.752, 177.833]</small></div>
  </div>
  <div class="result-row good-result">
    <div class="result-label"><b>4160 B</b><span>+16 int</span></div>
    <div class="result-track"><i style="width: 32.3%"></i></div>
    <div class="result-value"><b>56.598</b><span>μs</span><small>[55.957, 58.597]</small></div>
  </div>
</div>

<div class="speedup-number"><b>3.10×</b><span>faster</span></div>

<!--
口播提示：这里才是控制变量验证。padding 不参与求和，只改变每行的物理起点；括号里是 p10 和 p90。
-->

---

# 破案之后，还剩一个巧合

<div class="coincidence-layout">
  <div class="coincidence-card stride-card">
    <span>row stride</span>
    <b>1024 int × 4 B</b>
    <em>4096 B</em>
  </div>
  <div class="coincidence-equal">=</div>
  <div class="coincidence-card page-card">
    <span>x86-64 base page</span>
    <b>2¹² B</b>
    <em>4096 B</em>
  </div>
</div>

<div class="coincidence-bits">
  <span>page number</span><b>page offset [11:0]</b>
  <i>set [11:6]</i><em>line offset [5:0]</em>
</div>

<div class="question-chip">为什么 L1 set index 恰好不越过 bit 11？</div>

<!--
口播提示：这是第二幕的入口。4096 B 既是坏 stride，也是本机基础页大小；先把这个巧合留成问题。
-->

---

# 如果再借 4 个 Bits 呢？

<div class="bigger-index-layout">
  <div class="index-proposal">
    <div class="proposal-bits"><span>tag</span><b>set [15:6]</b><i>offset [5:0]</i></div>
    <div class="proposal-count">2¹⁰ = <b>1024 sets</b></div>
  </div>
  <div class="proposal-arrow">→</div>
  <div class="proposal-capacity">
    <span>1024 sets</span><i>×</i><span>8 ways</span><i>×</i><span>64 B</span>
    <b>512 KiB L1D?</b>
  </div>
</div>

<div class="proposal-blocker"><span>bits [15:12]</span><b>不再属于 4 KiB 页内偏移</b></div>

<!--
口播提示：这只是思想实验，不是说 512 KiB L1 就现实可行。它揭示真正的门槛：额外 index bits 需要地址翻译，或需要额外的 alias/page-colouring 机制。
-->

---

# 纯分段像停车位：旁边有人，就长不大

<div class="segment-page-compare">
  <div class="segment-side">
    <div class="memory-strip">
      <span class="seg-a">A</span><i>hole</i><span class="seg-b">B</span><i>hole</i><span class="seg-c">C</span>
    </div>
    <div class="growth-attempt"><b>A wants +3</b><i>→</i><em>blocked by B</em></div>
  </div>
  <div class="paging-side">
    <div class="page-routes">
      <div><span>A0</span><i>→</i><b>frame 2</b></div>
      <div><span>A1</span><i>→</i><b>frame 6</b></div>
      <div><span>A2</span><i>→</i><b>frame 0</b></div>
      <div><span>A3</span><i>→</i><b>frame 4</b></div>
    </div>
    <div class="physical-caption">physical frame order</div>
    <div class="physical-pages">
      <span><small>0</small>A2</span><b><small>1</small>B</b><span><small>2</small>A0</span><em><small>3</small>free</em>
      <span><small>4</small>A3</span><b><small>5</small>C</b><span><small>6</small>A1</span>
    </div>
  </div>
</div>

<div class="memory-model-table">
  <div class="table-head"><span></span><b>连续分段</b><b>分页</b></div>
  <div><span>进程眼中</span><em>连续区间</em><em>连续虚拟页</em></div>
  <div><span>物理内存</span><em>必须连续</em><em>页框可离散</em></div>
  <div><span>增长</span><em>需要相邻空洞</em><em>再映射一页</em></div>
</div>

<!--
口播提示：用停车位故事讲“为什么要分页”。这里讲的是连续物理分配的外部碎片和增长约束；现代 x86-64 仍保留分段机制，但通用用户空间主要依赖分页。
-->

---

# 翻译地址时，页内位置原封不动

<div class="translation-plot">
  <div class="address-row virtual-row"><b>Virtual address</b><span>VPN = 0x12345</span><i>offset = 0x678</i></div>
  <div class="translation-center">
    <div class="page-table-box">page table<br><b>0x12345 → 0x9abcd</b></div>
    <div class="offset-pass">offset unchanged</div>
  </div>
  <div class="address-row physical-row"><b>Physical address</b><span>PFN = 0x9abcd</span><i>offset = 0x678</i></div>
</div>

<div class="page-equation">VA = VPN │ offset &nbsp;&nbsp;→&nbsp;&nbsp; PA = PFN │ <b>same offset</b></div>

<!--
口播提示：4 KiB 页内 offset 是 12 bits。页表只替换高位页号，低 12 bits 在虚拟地址和物理地址中相同；这是 VIPT 能重叠工作的根基。
-->

---

# 页表很大：TLB 缓存“翻译结果”

<div class="tlb-path">
  <div class="tlb-request">VA<br><b>VPN</b></div>
  <div class="tlb-arrow">→</div>
  <div class="tlb-cache">
    <div class="tlb-title">dTLB</div>
    <div class="tlb-entry"><span>VPN</span><b>PFN</b></div>
    <div class="tlb-entry"><span>VPN</span><b>PFN</b></div>
    <div class="tlb-entry active"><span>0x12345</span><b>0x9abcd</b></div>
  </div>
  <div class="tlb-hit"><b>HIT</b><span>PFN ready</span><i>MISS →</i></div>
  <div class="page-walk">
    <div>PGD</div><i>→</i><div>P4D</div><i>→</i><div>PUD</div><i>→</i><div>PMD</div><i>→</i><div>PTE</div>
  </div>
</div>

<div class="tlb-distinction"><span>TLB miss</span><b>→ page-table walk</b><i>≠</i><span>page fault</span></div>

<!--
口播提示：TLB 是页表翻译缓存。TLB miss 通常触发硬件 page walk；只有翻译不存在、页面不在内存或权限不允许等情况才进入 page-fault 处理。
来源：https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
https://docs.kernel.org/mm/page_tables.html
-->

---

# Page Fault 不等于 Segmentation Fault

<div class="fault-table">
  <div class="fault-head"><span>状态</span><span>硬件 / 内核动作</span><span>结果</span></div>
  <div class="fault-row good"><b>映射有效 · 页面在场</b><span>完成翻译</span><em>继续执行</em></div>
  <div class="fault-row normal"><b>允许访问 · 尚未映射</b><span>lazy allocation / demand paging</span><em>处理后重试</em></div>
  <div class="fault-row normal"><b>可写 VMA · PTE 只读</b><span>Copy-on-Write</span><em>复制后重试</em></div>
  <div class="fault-row bad"><b>无效地址 / 权限违反</b><span>无法合法修复</span><em>SIGSEGV</em></div>
</div>

<div class="fault-caption">page fault 是异常入口；SIGSEGV 只是其中一种用户态结局</div>

<!--
口播提示：可以讲“内核判官”的故事。正常的 lazy allocation 和 COW 都故意借 page fault 完成；非法访问才通常收到 SIGSEGV。
来源：https://docs.kernel.org/mm/page_tables.html
-->

---

# 内核的三次“偷天换日”

<div class="kernel-tricks">
  <div class="kernel-trick">
    <div class="trick-title">共享只读代码页</div>
    <div class="shared-code-map"><span>ls #1 VA</span><i>→</i><b>one text frame</b><i>←</i><span>ls #2 VA</span></div>
    <em>一份物理代码，多进程映射</em>
  </div>
  <div class="kernel-trick">
    <div class="trick-title">Copy-on-Write</div>
    <div class="cow-flow"><span>shared RO</span><i>write fault</i><b>private copy</b></div>
    <em>写之前共享，写时才复制</em>
  </div>
  <div class="kernel-trick">
    <div class="trick-title">Demand Paging</div>
    <div class="demand-flow"><span>VMA reserved</span><i>first touch</i><b>map page</b></div>
    <em>真正访问时才建立在场页</em>
  </div>
</div>

<!--
口播提示：三张图支撑新版故事：文件支持的只读代码页可共享；fork 常通过 COW 延迟复制；页面也可按需进入内存。具体装载与预读策略会因内核和文件系统而异，不要说“exec 后所有页表必定全空”。
-->

---

# 同一个物理页，可能有两个虚拟名字

<div class="alias-layout">
  <div class="virtual-alias va-a"><b>VA A</b><span>0x12345│678</span></div>
  <div class="virtual-alias va-b"><b>VA B</b><span>0xfedcb│678</span></div>
  <div class="alias-arrows"><i>↘</i><i>↙</i></div>
  <div class="physical-frame"><b>one physical frame</b><span>PFN 0x9abcd</span></div>
  <div class="viva-problem">
    <span>若 cache 用完整虚拟地址认人</span>
    <div><i>cache copy A</i><i>cache copy B</i></div>
    <b>synonym / coherence problem</b>
  </div>
</div>

<!--
口播提示：这就是 VIVT 的 synonym 问题：两个虚拟地址指向同一物理数据，却可能进入两个 cache entries。读写一致性、失效和上下文切换都更难处理。
-->

---

# 三种 Cache 地址策略

<div class="cache-strategy-table">
  <div class="strategy-head"><span></span><b>Index</b><b>Tag</b><b>与 TLB 重叠</b><b>别名处理</b></div>
  <div class="strategy-row vivt"><strong>VIVT</strong><span>Virtual</span><span>Virtual</span><em>最好</em><i>困难</i></div>
  <div class="strategy-row pipt"><strong>PIPT</strong><span>Physical</span><span>Physical</span><em>需先翻译</em><i>简单</i></div>
  <div class="strategy-row vipt"><strong>VIPT</strong><span>Virtual</span><span>Physical</span><em>可重叠</em><i>受 index bits 约束</i></div>
</div>

<div class="strategy-winner"><b>VIPT</b><span>先用虚拟地址选 set，再用物理 tag 认人</span></div>

<!--
口播提示：这是教学对比表。VIVT 查询快但有 homonym/synonym 管理成本；PIPT 物理唯一性最直接；VIPT 折中，让 index lookup 与翻译重叠。
来源：https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/page-colouring-on-armv6-and-a-bit-on-armv7
https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
-->

---

# VIPT：dTLB 与 L1D 同时开工

<div class="vipt-timeline">
  <div class="timeline-axis"><div class="timeline-ticks"><span>t₀</span><span>t₁</span><span>t₂</span><span>t₃</span><span>t₄</span></div></div>
  <div class="timeline-row"><b>dTLB</b><span class="phase translate">VPN → PFN</span><i></i><i></i></div>
  <div class="timeline-row"><b>L1D</b><span class="phase select">VA[11:6] selects set</span><span class="phase readways">read 8 ways</span><i></i></div>
  <div class="timeline-row"><b>merge</b><i></i><i></i><span class="phase compare">physical tag compare</span><span class="phase output">data</span></div>
</div>

<div class="parallel-brace"><span>dTLB translation</span><b>∥</b><span>set lookup</span></div>

<!--
口播提示：这是依赖关系图，不是本机逐周期测量。虚拟 set index 先取出候选 ways；TLB 同时产生物理页号；最后用 physical tag 并行比较并选数据。
-->

---

# 4 KiB 页给“无别名 VIPT”多少 Index 空间？

<div class="vipt-budget">
  <div class="budget-address"><span>translated page number</span><b>set: 6 bits</b><i>offset: 6 bits</i></div>
  <div class="budget-brace"><span>4 KiB page offset = 12 bits</span></div>
</div>

<div class="budget-equations">
  <div><span>sets × line size</span><b>≤ page size</b></div>
  <div><span>cache size</span><b>≤ page size × ways</b></div>
</div>

<div class="host-budget"><span>4 KiB</span><i>×</i><span>8 ways</span><i>=</i><b>32 KiB L1D</b></div>

<!--
口播提示：在 index 完全位于 page offset、无需额外 alias 管理的简化 VIPT 设计里，每个 way 的索引跨度不能超过页大小，所以总容量上界是 page size × ways。对本机正好是 4 KiB × 8 = 32 KiB。
-->

---

# 这不是 ISA 的铁律，而是设计取舍

<div class="design-table">
  <div class="design-head"><span>想扩大 L1</span><span>得到什么</span><span>要付什么</span></div>
  <div><b>增加 ways</b><span>同样 sets，容量变大</span><em>更多 tag 比较、功耗与布线</em></div>
  <div><b>更大基础页</b><span>更多 page-offset index bits</span><em>内存浪费与 OS/ABI 取舍</em></div>
  <div><b>跨页 offset 索引</b><span>更多 sets</span><em>page colouring / alias 管理</em></div>
  <div><b>PIPT / 其他微架构</b><span>摆脱简单 VIPT 上界</span><em>翻译依赖或额外硬件复杂度</em></div>
</div>

<div class="architecture-note">Arm 也存在 VIPT、PIPT 与 page-colouring 取舍；不能仅凭 ISA 名称推断缓存层级。</div>

<!--
口播提示：用这张表替代草稿里不确定的 Apple/ARM 容量故事。Arm 官方资料本身就展示了 VIPT、PIPT 和 page-colouring 的多种选择；具体缓存层级属于具体芯片微架构，不是“Arm 天生没有这个烦恼”。
来源：https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/page-colouring-on-armv6-and-a-bit-on-armv7
-->

---

# CPU 也有 “Bank Conflict” 吗？

<div class="analogy-visual">
  <div class="analogy-side gpu-side">
    <div class="analogy-title">GPU</div>
    <div class="bank-row"><i></i><i></i><i class="hot"></i><i></i><i></i><i></i><i></i><i></i></div>
    <b>banks</b>
    <span>同一条指令争同一 bank</span>
  </div>
  <div class="not-equal">≠<small>同一机制</small></div>
  <div class="analogy-side cpu-side">
    <div class="analogy-title">CPU</div>
    <div class="cache-set-icon"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><em>+120</em></div>
    <b>sets × ways</b>
    <span>不同 lines 争同一 set</span>
  </div>
</div>

<div class="shared-pattern">regular stride <b>→</b> too few hardware buckets</div>

<!--
口播提示：直觉相似，标准名称不同。GPU 是 bank conflict；这里是 cache set conflict 或 conflict miss。
-->

---

# 怎么修？

<div class="fix-visuals">
  <div class="fix-card primary">
    <div class="fix-icon loop-icon"><span>→→→→</span><span>→→→→</span><span>→→→→</span></div>
    <b>交换循环</b>
    <code>j 放内层</code>
  </div>
  <div class="fix-card">
    <div class="fix-icon tile-icon"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
    <b>Blocking</b>
    <code>先吃完一块</code>
  </div>
  <div class="fix-card">
    <div class="fix-icon pad-icon"><span>1024</span><i>+</i><b>16</b></div>
    <b>Padding / Prepack</b>
    <code>打破规则步长</code>
  </div>
</div>

<!--
口播提示：能改算法时优先连续遍历，其次 blocking。必须垂直访问、且会复用数据时，再测 padding 或 prepack 的收益。
-->

---

<!-- _class: closing -->
<!-- _paginate: false -->
<!-- _footer: "" -->

<div class="closing-story">
  <span>4 KiB pages</span><b>→</b><span>VIPT index budget</span><b>→</b><span>64 sets</span><b>→</b><span>4096 B resonance</span>
</div>

<div class="closing-equation"><span>奇怪的行为</span><b>=</b><span>工程取舍留下的伤疤</span></div>

<!--
口播提示：结尾把两幕扣回同一条链：分页提供了虚拟内存与共享，VIPT 把翻译藏进 L1 lookup，但 index budget 又留下 4096 B stride resonance。软硬件工程师是在物理、功耗、面积、兼容性之间做取舍。
-->
