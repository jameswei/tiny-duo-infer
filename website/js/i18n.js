/**
 * i18n.js — EN / ZH language switch for tiny-duo-infer landing page.
 *
 * Usage:
 *   - Plain-text elements: <tag data-i18n="key">fallback</tag>
 *   - Elements with inner HTML (links, <code>): <tag data-i18n-html="key">fallback</tag>
 *   - SVG <text> nodes: <text data-i18n="key">fallback</text>
 *   - Language preference is persisted in localStorage under "tdi-lang".
 */

const translations = {
  en: {
    /* ── Nav ── */
    "nav.about":    "About",
    "nav.features": "Features",
    "nav.stack":    "Stack",
    "nav.cli":      "CLI",
    "nav.lang":     "中文",

    /* ── Hero ── */
    "hero.tagline":     "A learning-first LLM inference engine built from scratch on Apple Silicon",
    "hero.cta_primary": "View on GitHub",
    "hero.cta_secondary": "Learn more ↓",

    /* ── About ── */
    "about.heading": "About",
    "about.p1": "tiny-duo-infer is a pure-Python LLM inference engine that teaches how large language models work under the hood — from tokenization to the final sampled token. Every inference concept is visible in readable code rather than hidden behind Transformers, mlx-lm, or vLLM.",
    "about.p2": "Each building block — prefill, decode, KV cache updates, grouped-query attention, RoPE rotary embeddings, SwiGLU feed-forward networks — is explicitly implemented so that you can trace exactly what happens between a prompt and a generated token.",
    "about.p3": "Inspired by nano-vllm and tiny-llm, the project follows a progressive learning roadmap: from single-user Llama inference to multi-model support, HTTP serving, per-request observability, and INT4/INT8 weight-only quantization.",
    "about.link": "Explore the source on GitHub →",

    /* ── Pipeline diagram ── */
    "diagram.prompt":     "Prompt",
    "diagram.tokenizer":  "Tokenizer",
    "diagram.engine":     "Prefill · Decode",
    "diagram.engine_sub": "Engine",
    "diagram.kv_cache":   "KV Cache",
    "diagram.sampler":    "Sampler",
    "diagram.output":     "Output",

    /* ── Six concepts ── */
    "concepts.intro":   "Read the code. Understand the engine. Six core inference concepts — fully explicit, no black boxes.",
    "concept.1.title":  "Prefill & Decode Loop",
    "concept.1.desc":   "Trace every token generation step — from KV cache population through the decode loop to EOS detection.",
    "concept.2.title":  "Grouped-Query Attention (GQA)",
    "concept.2.desc":   "See how K/V heads are expanded to match Q heads, and how the causal mask is built and applied at each step.",
    "concept.3.title":  "Rotary Position Embeddings (RoPE)",
    "concept.3.desc":   "Follow frequency precomputation and the explicit rotation formula applied pair-wise across the head dimension.",
    "concept.4.title":  "KV Cache Management",
    "concept.4.desc":   "Understand the update() / advance() protocol that keeps K/V positions consistent across all transformer layers.",
    "concept.5.title":  "Weight-Only Quantization",
    "concept.5.desc":   "Learn how INT4/INT8 quantized matmul replaces full-precision forward passes, and how memory savings are measured per run.",
    "concept.6.title":  "Sampling Strategies",
    "concept.6.desc":   "Step through greedy argmax, temperature scaling, top-k filtering, and top-p nucleus selection — all in plain Python.",

    /* ── Stats ── */
    "stats.phases":       "CORE CONCEPTS BUILT FROM SCRATCH",
    "stats.phases_sub":   "RoPE · GQA · KV Cache · SwiGLU · Quantization · ···",
    "stats.models":       "MODEL FAMILIES",
    "stats.models_sub":   "Llama-3.2-1B · Qwen3-0.6B",
    "stats.tests":        "TESTS PASSING",
    "stats.tests_sub":    "unit · integration · serving · quantization",
    "stats.wrappers":     "FRAMEWORK WRAPPERS",
    "stats.wrappers_sub": "no mlx-lm, no transformers core, pure Python",

    /* ── Capabilities ── */
    "cap.heading": "Capabilities",
    "cap.1.title": "Explicit Prefill → Decode Pipeline",
    "cap.1.desc":  "The full generation loop — prefill, KV cache population, decode step, EOS detection — is implemented in readable Python with no framework wrapper.",
    "cap.2.title": "GQA · RoPE · SwiGLU from Scratch",
    "cap.2.desc":  "Grouped-query attention, rotary position embeddings, and SwiGLU feed-forward networks are each implemented explicitly so every computation is traceable.",
    "cap.3.title": "INT4 / INT8 Weight-Only Quantization",
    "cap.3.desc":  "MLX-native quantized matmul with per-group affine scales. Memory accounting reports full-precision vs runtime linear-weight bytes for every run.",
    "cap.4.title": "HTTP Serving with NDJSON Streaming",
    "cap.4.desc":  "FastAPI server exposes /generate (JSON) and /generate/stream (NDJSON). Full 26-field stats included in every response.",
    "cap.5.title": "Per-request Observability",
    "cap.5.desc":  "Time-to-first-token, decode throughput, KV-cache memory, context-budget policy outcome, and quantization stats — reported on every request via CLI or HTTP.",
    "cap.6.title": "Multi-model Portability",
    "cap.6.desc":  "Supports Llama-3.2-1B and Qwen3-0.6B on the same engine, demonstrating model-family portability before backend portability.",

    /* ── Stack ── */
    "stack.heading":    "Tech Stack",
    "stack.lang":       "Language",
    "stack.backend":    "Backend",
    "stack.models_row": "Models",
    "stack.tokenizer":  "Tokenizer",
    "stack.weights":    "Weights",
    "stack.serving":    "Serving",
    "stack.tooling":    "Tooling",

    /* ── CLI ── */
    "cli.heading":   "CLI & API",
    "cli.desc":      "Run inference locally from the command line, start the HTTP server, or profile generation across quantization modes — all with a single uv run.",
    "cli.docs_link": "Full documentation on GitHub →",

    /* ── CTA ── */
    "cta.heading": "Interested in the source?",
    "cta.button":  "View on GitHub",

    /* ── Footer ── */
    "footer.built_by": "Built by <a href=\"https://linkedin.com/in/jameswei\" target=\"_blank\" rel=\"noopener\">James Wei</a>",
    "footer.github":   "GitHub",
    "footer.linkedin": "LinkedIn",
    "footer.repo":     "Repository",
  },

  zh: {
    /* ── Nav ── */
    "nav.about":    "关于",
    "nav.features": "特性",
    "nav.stack":    "技术栈",
    "nav.cli":      "命令行",
    "nav.lang":     "English",

    /* ── Hero ── */
    "hero.tagline":       "在 Apple Silicon 上从零构建的学习型 LLM 推理引擎",
    "hero.cta_primary":   "在 GitHub 上查看",
    "hero.cta_secondary": "了解更多 ↓",

    /* ── About ── */
    "about.heading": "关于项目",
    "about.p1": "tiny-duo-infer 是一个纯 Python 实现的 LLM 推理引擎，旨在通过从零构建的方式深入理解大型语言模型的工作原理——从分词到最终采样的 token，每一个推理环节都清晰可见，而非隐藏在 Transformers、mlx-lm 或 vLLM 的黑盒之后。",
    "about.p2": "每个核心组件——prefill、decode、KV cache 更新、分组查询注意力（GQA）、旋转位置编码（RoPE）、SwiGLU 前馈网络——均经过显式实现，你可以完整追踪从 prompt 到生成 token 之间发生的每一步计算。",
    "about.p3": "项目受 nano-vllm 和 tiny-llm 启发，采用渐进式学习路线：从单用户 Llama 推理出发，逐步扩展至多模型支持、HTTP 服务、单次请求可观测性，以及 INT4/INT8 仅权重量化。",
    "about.link": "在 GitHub 上探索源码 →",

    /* ── Pipeline diagram ── */
    "diagram.prompt":     "输入",
    "diagram.tokenizer":  "分词器",
    "diagram.engine":     "Prefill · Decode",
    "diagram.engine_sub": "推理引擎",
    "diagram.kv_cache":   "KV 缓存",
    "diagram.sampler":    "采样器",
    "diagram.output":     "输出",

    /* ── Six concepts ── */
    "concepts.intro":   "读代码，懂引擎。六个核心推理概念——完全显式，零黑盒。",
    "concept.1.title":  "Prefill & Decode 循环",
    "concept.1.desc":   "追踪每一步 token 生成过程——从 KV cache 填充，到 decode 循环，再到 EOS 检测。",
    "concept.2.title":  "分组查询注意力（GQA）",
    "concept.2.desc":   "了解 K/V head 如何扩展以匹配 Q head，以及因果掩码如何在每一步被构建和应用。",
    "concept.3.title":  "旋转位置编码（RoPE）",
    "concept.3.desc":   "追踪频率预计算过程，以及在 head 维度逐对应用的显式旋转公式。",
    "concept.4.title":  "KV Cache 管理",
    "concept.4.desc":   "理解 update() / advance() 协议如何在所有 transformer 层中保持 K/V 位置的一致性。",
    "concept.5.title":  "仅权重量化",
    "concept.5.desc":   "了解 INT4/INT8 量化矩阵乘如何替代全精度前向传播，以及每次运行如何测量内存节省。",
    "concept.6.title":  "采样策略",
    "concept.6.desc":   "逐步了解 greedy argmax、温度缩放、top-k 过滤和 top-p 核采样——全部用纯 Python 实现。",

    /* ── Stats ── */
    "stats.phases":       "从零构建的核心概念数",
    "stats.phases_sub":   "RoPE · GQA · KV Cache · SwiGLU · 量化 · ···",
    "stats.models":       "支持模型系列",
    "stats.models_sub":   "Llama-3.2-1B · Qwen3-0.6B",
    "stats.tests":        "测试用例通过",
    "stats.tests_sub":    "单元 · 集成 · 服务 · 量化",
    "stats.wrappers":     "框架封装层",
    "stats.wrappers_sub": "无 mlx-lm，无 transformers 核心，纯 Python",

    /* ── Capabilities ── */
    "cap.heading": "核心特性",
    "cap.1.title": "显式 Prefill → Decode 流水线",
    "cap.1.desc":  "完整的生成循环——prefill、KV cache 填充、decode 步骤、EOS 检测——均以可读的 Python 代码实现，无任何框架封装。",
    "cap.2.title": "从零实现 GQA · RoPE · SwiGLU",
    "cap.2.desc":  "分组查询注意力、旋转位置编码和 SwiGLU 前馈网络均经过显式实现，每一步计算均可追踪。",
    "cap.3.title": "INT4 / INT8 仅权重量化",
    "cap.3.desc":  "基于 MLX 原生量化矩阵乘法，支持分组仿射缩放。内存统计报告每次运行的全精度与运行时线性层权重字节数对比。",
    "cap.4.title": "HTTP 服务与 NDJSON 流式传输",
    "cap.4.desc":  "FastAPI 服务暴露 /generate（JSON）和 /generate/stream（NDJSON）接口，每次响应均包含完整的 26 字段统计信息。",
    "cap.5.title": "单次请求可观测性",
    "cap.5.desc":  "首 token 时间（TTFT）、decode 吞吐量、KV cache 内存、上下文预算策略结果及量化统计——通过 CLI 或 HTTP 对每次请求进行上报。",
    "cap.6.title": "多模型可移植性",
    "cap.6.desc":  "同一推理引擎支持 Llama-3.2-1B 和 Qwen3-0.6B，在扩展后端可移植性之前率先验证模型族可移植性。",

    /* ── Stack ── */
    "stack.heading":    "技术栈",
    "stack.lang":       "语言",
    "stack.backend":    "后端",
    "stack.models_row": "模型",
    "stack.tokenizer":  "分词器",
    "stack.weights":    "权重加载",
    "stack.serving":    "服务",
    "stack.tooling":    "工具链",

    /* ── CLI ── */
    "cli.heading":   "命令行与 API",
    "cli.desc":      "通过命令行在本地运行推理、启动 HTTP 服务，或跨量化模式进行生成性能剖析——均通过单条 uv run 命令完成。",
    "cli.docs_link": "在 GitHub 上查看完整文档 →",

    /* ── CTA ── */
    "cta.heading": "对源码感兴趣？",
    "cta.button":  "在 GitHub 上查看",

    /* ── Footer ── */
    "footer.built_by": "由 <a href=\"https://linkedin.com/in/jameswei\" target=\"_blank\" rel=\"noopener\">James Wei</a> 构建",
    "footer.github":   "GitHub",
    "footer.linkedin": "LinkedIn",
    "footer.repo":     "代码仓库",
  },
};

/* ── Language application ── */

function setLanguage(lang) {
  const t = translations[lang];
  if (!t) return;

  /* Plain-text elements */
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (t[key] !== undefined) el.textContent = t[key];
  });

  /* Elements with inner HTML (links, <code>, etc.) */
  document.querySelectorAll("[data-i18n-html]").forEach((el) => {
    const key = el.getAttribute("data-i18n-html");
    if (t[key] !== undefined) el.innerHTML = t[key];
  });

  /* Page title */
  document.title =
    lang === "zh"
      ? "tiny-duo-infer — 学习型 LLM 推理引擎"
      : "tiny-duo-infer — Learning-first LLM Inference Engine";

  /* html lang attribute */
  document.documentElement.lang = lang;

  /* Persist preference */
  try { localStorage.setItem("tdi-lang", lang); } catch (_) {}
}

function toggleLang() {
  const current = document.documentElement.lang || "en";
  setLanguage(current === "en" ? "zh" : "en");
}

/* ── Init: apply stored or browser preference ── */
(function init() {
  let lang = "en";
  try { lang = localStorage.getItem("tdi-lang") || lang; } catch (_) {}

  /* Fall back to browser language if no stored preference */
  if (!localStorage.getItem("tdi-lang")) {
    const browser = (navigator.language || "en").toLowerCase();
    if (browser.startsWith("zh")) lang = "zh";
  }

  setLanguage(lang);
})();
