#!/usr/bin/env python3
"""
本地路由器 - 轻量级本地模型路由系统
48GB VRAM 双模型配置（Qwen3.6 + Gemma4）

版本: v2.0（Qwen3.6-35B-A3B + Gemma4-31B）
创建日期: 2026-04-16
更新日期: 2026-04-18
作者: 忒弥斯 (T-Mind) 🔮

核心策略:
- 编程常驻：Qwen3.6-35B-A3B（MoE，只激活 3B，~22GB nvfp4）
- 多模态按需：Gemma4-31B（dense 31B，~18.5GB）
- 48GB VRAM：可同时加载，32K ctx
"""

import logging
import re
import time
from typing import Optional, Tuple

import ollama

# ── 输入长度限制 ──────────────────────────────────────────
MAX_PROMPT_CHARS = 8000  # ~2K tokens，防止 OOM

logger = logging.getLogger(__name__)


class LocalRouter:
    """本地模型路由器 - 意图识别 + 模型调度"""

    # 模型配置（v2.0: Qwen3.6 + Gemma4, 48GB VRAM）
    MODELS = {
        "coding": "qwen3.6:35b-a3b-coding-nvfp4",     # 编程核心，MoE 3B active，常驻
        "thinking": "qwen3.6:35b-a3b",                  # 推理 + thinking mode
        "vision": "gemma4:31b",                          # 多模态（文本+图片），dense 31B
        "tools": "gemma4:26b",                           # 工具调用 / agentic，MoE 4B active
        "fast": "qwen3.6:35b-a3b",                       # 快速响应，MoE 3B active
        "fallback": "qwen3.6:35b-a3b"                    # 降级方案
    }

    def __init__(self, check_connection: bool = True):
        """初始化路由器

        Args:
            check_connection: 是否检查 Ollama 连接（默认 True）
        """
        self.current_model = None
        self.stats = {
            "total_requests": 0,
            "coding_requests": 0,
            "tools_requests": 0,
            "fallback_requests": 0
        }
        self._ollama_available = True

        if check_connection:
            self._check_ollama()

    def _check_ollama(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            models = ollama.list()
            available = [m.get("model", "") or m.get("name", "") for m in models.get("models", [])]
            logger.info(f"Ollama 可用，已加载 {len(available)} 个模型")
            self._ollama_available = True
            return True
        except Exception as e:
            logger.warning(f"Ollama 不可用: {e}")
            self._ollama_available = False
            return False

    # 意图识别规则（关键词匹配）
    INTENT_PATTERNS = {
        "coding": [
            "写代码", "编程", "函数", "类", "debug", "调试", "重构",
            "算法", "数据结构", "实现", "开发", "python", "javascript",
            "代码", "程序", "接口", "模块", "方法"
        ],
        "tools": [
            "调用", "api", "文件", "命令", "工具", "执行", "运行",
            "系统", "脚本", "自动化", "批处理", "shell"
        ]
    }

    # ── 专用系统提示词模板 ──────────────────────────────────
    SYSTEM_PROMPTS = {
        "coding": (
            "你是一个专业的程序员助手。请严格遵守以下规则：\n"
            "1. 只使用该语言的标准库和标准语法。\n"
            "2. 绝对不要编造、猜测或幻想任何不存在的函数、方法、模块或类。"
            "如果你不确定某个方法是否存在，就不要使用它。\n"
            "3. list 没有 .partition()、.split()、.find()、.replace() 方法。\n"
            "4. dict 没有 .sort() 方法；set 没有 .append() 方法。\n"
            "5. str 的 .split() 返回 list，list 的 .split() 不存在。\n"
            "6. 如果不确定，写出注释说明并给出替代方案。\n"
            "7. 代码必须有注释。回答使用中文。"
        ),
        "tools": (
            "你是一个系统工具调用助手。请严格遵守以下规则：\n"
            "1. 只推荐你确定存在的命令和工具。\n"
            "2. 不要猜测命令参数，只使用你确定正确的参数。\n"
            "3. 如果不确定，明确说明并建议查阅文档。\n"
            "4. 回答使用中文。"
        ),
        "general": (
            "你是一个有用的助手。请严格遵守以下规则：\n"
            "1. 只回答你确定的事实。如果不确定，明确说明。\n"
            "2. 不要编造函数、方法、库或 API。\n"
            "3. 回答使用中文。"
        ),
    }

    # ── 幻觉检测模式 ──────────────────────────────────────
    HALLUCINATION_PATTERNS = [
        # list 上不存在的方法
        (r'\blist\.\s*(partition|split|find|replace|strip|encode|decode|join|startswith|endswith|index\s*\()\b',
         "list 没有 .{method}() 方法"),
        # dict 上不存在的方法
        (r'\bdict\.\s*(sort|append|remove|index)\b',
         "dict 没有 .{method}() 方法"),
        # set 上不存在的方法
        (r'\bset\.\s*(append|sort|index|insert)\b',
         "set 没有 .{method}() 方法"),
        # tuple 上不存在的方法
        (r'\btuple\.\s*(append|sort|insert|remove|extend|pop)\b',
         "tuple 没有 .{method}() 方法"),
        # int/float 上不存在的方法
        (r'\b(int|float)\.\s*(split|join|strip|find|replace)\b',
         "{type} 没有 .{method}() 方法"),
    ]

    def route(self, prompt: str) -> str:
        """
        意图识别 + 路由决策

        Args:
            prompt: 用户输入

        Returns:
            模型名称
        """
        prompt_lower = prompt.lower()

        # 编程任务优先（核心场景）
        for keyword in self.INTENT_PATTERNS["coding"]:
            if keyword in prompt_lower:
                self.stats["coding_requests"] += 1
                return self.MODELS["coding"]

        # 工具调用
        for keyword in self.INTENT_PATTERNS["tools"]:
            if keyword in prompt_lower:
                self.stats["tools_requests"] += 1
                return self.MODELS["tools"]

        # 默认使用编程模型（适合企业软件开发）
        self.stats["fallback_requests"] += 1
        return self.MODELS["fallback"]

    @classmethod
    def _detect_task_type(cls, prompt: str) -> str:
        """根据关键词判断任务类型，返回对应的提示词 key。
        复用 INTENT_PATTERNS，与 route() 保持一致。
        """
        p = prompt.lower()
        for kw in cls.INTENT_PATTERNS["coding"]:
            if kw in p:
                return "coding"
        for kw in cls.INTENT_PATTERNS["tools"]:
            if kw in p:
                return "tools"
        return "general"

    def _postprocess(self, text: str) -> str:
        """输出后处理：检测常见幻觉模式，添加警告"""
        warnings = []
        for pattern, msg_template in self.HALLUCINATION_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                warnings.append(msg_template)
        if warnings:
            unique = list(dict.fromkeys(warnings))  # 去重保序
            text += "\n\n⚠️ [幻觉检测警告] " + "；".join(unique)
        return text

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, float, str]:
        """
        统一生成接口

        Args:
            prompt: 用户输入
            model: 指定模型（可选，覆盖路由决策）
            system_prompt: 系统提示词（可选，覆盖自动选择）

        Returns:
            (响应文本, 延迟秒数, 使用的模型)
        """
        self.stats["total_requests"] += 1
        start_time = time.time()

        # ── 输入长度限制 ──
        if len(prompt) > MAX_PROMPT_CHARS:
            prompt = prompt[:MAX_PROMPT_CHARS] + "\n\n[输入已截断，超出长度限制]"

        # 路由决策
        if model is None:
            model = self.route(prompt)

        # ── 系统提示词选择（优先级：显式传入 > 自动检测） ──
        if system_prompt:
            effective_system = system_prompt
        else:
            task_type = self._detect_task_type(prompt)
            effective_system = self.SYSTEM_PROMPTS.get(task_type, self.SYSTEM_PROMPTS["general"])

        full_prompt = f"{effective_system}\n\n{prompt}"

        # 调用 Ollama API
        try:
            response = ollama.generate(model=model, prompt=full_prompt)
            result = response.get("response", "")

            # 清理 DeepSeek-R1 的思考过程（多种 think 标签格式）
            result = re.sub(r'<think[^>]*>.*?</think\s*>', '', result, flags=re.DOTALL).strip()
            result = re.sub(r'<think.*$', '', result, flags=re.DOTALL).strip()

            # ── 输出后处理：幻觉检测 ──
            result = self._postprocess(result)

            latency = time.time() - start_time
            return result, latency, model

        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"错误: {str(e)}"
            return error_msg, latency, model

    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        流式生成接口（用于 CLI 逐字打印）

        Args:
            prompt: 用户输入
            model: 指定模型（可选）
            system_prompt: 系统提示词（可选）

        Yields:
            (chunk_text, is_done) — is_done=True 时生成结束
        """
        self.stats["total_requests"] += 1
        start_time = time.time()

        if len(prompt) > MAX_PROMPT_CHARS:
            prompt = prompt[:MAX_PROMPT_CHARS] + "\n\n[输入已截断，超出长度限制]"

        if model is None:
            model = self.route(prompt)

        if system_prompt:
            effective_system = system_prompt
        else:
            task_type = self._detect_task_type(prompt)
            effective_system = self.SYSTEM_PROMPTS.get(task_type, self.SYSTEM_PROMPTS["general"])

        messages = [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": prompt},
        ]

        try:
            stream = ollama.chat(model=model, messages=messages, stream=True)
            full_text = ""
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    full_text += content
                    yield content, False

            # 清理 think 标签
            full_text = re.sub(r'<think[^>]*>.*?</think\s*>', '', full_text, flags=re.DOTALL).strip()
            full_text = re.sub(r'<think.*$', '', full_text, flags=re.DOTALL).strip()
            full_text = self._postprocess(full_text)

            latency = time.time() - start_time
            yield full_text, True  # 最终结果 + 完成信号
            # Store last result for stats
            self._last_stream = (full_text, latency, model)

        except Exception as e:
            latency = time.time() - start_time
            self._last_stream = (f"错误: {e}", latency, model)
            yield f"错误: {e}", True

    def get_stats(self) -> dict:
        """获取统计信息"""
        if self.stats["total_requests"] == 0:
            return self.stats

        return {
            **self.stats,
            "coding_ratio": f"{self.stats['coding_requests'] / self.stats['total_requests'] * 100:.1f}%",
            "tools_ratio": f"{self.stats['tools_requests'] / self.stats['total_requests'] * 100:.1f}%",
            "fallback_ratio": f"{self.stats['fallback_requests'] / self.stats['total_requests'] * 100:.1f}%"
        }


def main():
    """命令行界面 — 流式输出 + Rich 思考动画"""
    router = LocalRouter()

    # Rich 可选依赖
    try:
        from rich.console import Console
        console = Console()
        _has_rich = True
    except ImportError:
        _has_rich = False

    print("🔮 本地路由器 v1.2 - 流式输出 + Rich 动画")
    print("=" * 50)
    print("已加载模型:")
    print("  - 编程: DeepSeek-R1 8B (4.9GB)")
    print("  - 工具: Gemma3-Tools 12B (8.1GB)")
    print("  - 内存优化: 单模型 + 按需切换")
    print("\n输入 'quit' 退出，'stats' 查看统计")
    print("=" * 50)

    while True:
        try:
            prompt = input("\n🤔 你: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ["quit", "exit", "q"]:
                print("\n👋 再见！")
                break

            if prompt.lower() == "stats":
                stats = router.get_stats()
                print("\n📊 统计信息:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue

            # ── 流式生成 ──
            # Rich spinner（思考动画）
            if _has_rich:
                console.print()
                with console.status("[bold green]🤔 思考中...", spinner="dots"):
                    first_chunk = True
                    for chunk, done in router.generate_stream(prompt):
                        if done:
                            final_text, latency, model = chunk, *router._last_stream[1:]
                            break
                        if first_chunk:
                            console.print()  # spinner 自动隐藏
                            first_chunk = False
                        print(chunk, end="", flush=True)
                latency = router._last_stream[1]
                model = router._last_stream[2]
                final_text = router._last_stream[0]
            else:
                print("\n🤖 思考中...")
                first_chunk = True
                for chunk, done in router.generate_stream(prompt):
                    if done:
                        final_text, latency, model = router._last_stream
                        break
                    if first_chunk:
                        print()
                        first_chunk = False
                    print(chunk, end="", flush=True)

            print()
            print(f"\n✅ 完成 ({latency:.2f}s, {model})")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()
