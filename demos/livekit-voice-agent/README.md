# LiveKit Voice Agent —— 非双工(turn-based)语音对照实现

一个用 [LiveKit Agents](https://github.com/livekit/agents) 框架搭的实时语音助手 "Kelly",
在 **macOS(Apple Silicon)本地** 跑通,终端里用麦克风直接对话。

这是 ITC 里 **turn-based(非双工)** 范式的对照参考:你说完 → 模型处理 → 模型回答,一问一答。
与之对照的全双工实现见 [`../minicpm-o-4.5-fullduplex`](../minicpm-o-4.5-fullduplex)。

## 它是什么 / 不是什么

- LiveKit Agents 是一个**搭语音 agent 的框架**,本身不含模型,需要外接 STT(耳朵)/ LLM(脑子)/ TTS(嘴)。
- 官方 `basic_agent.py` 用 LiveKit 托管推理(Deepgram + Cartesia + OpenAI,需 LiveKit Cloud 凭证)。
- **本 demo 改用 OpenAI Realtime 模型**:一个 `OPENAI_API_KEY` 同时搞定 STT+LLM+TTS,
  不需要 LiveKit 服务器、不需要额外的 Deepgram/Cartesia key。`console` 模式在本地终端跑,带麦克风/扬声器。

## 环境要求

- macOS(Apple Silicon 验证过 M4 Max)
- **Python 3.12**(⚠️ livekit-agents 暂不支持 3.14)。推荐用 [`uv`](https://docs.astral.sh/uv/) 管理。
- 一个 OpenAI API key

## 跑通步骤

```bash
# 1. 装 uv(若未装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 建 3.12 虚拟环境并装依赖
cd demos/livekit-voice-agent
uv venv --python 3.12
uv pip install -r requirements.txt

# 3. 配置 key
cp .env.example .env
#   编辑 .env,把 OPENAI_API_KEY 换成你的

# 4. 在「终端」里运行(需要真实 TTY + 麦克风权限)
uv run python agent.py console
```

运行后直接对着电脑说话(中英文都行),`Ctrl+C` 退出。
首次 macOS 会弹窗请求**麦克风权限**,点允许。

> macOS 用户也可以直接**双击** `启动语音助手.command`(可能需右键→打开以绕过 Gatekeeper)。

## 验证记录

实测启动 ~20 秒内即:连上 OpenAI Realtime API → 生成开场白
("Hi there! I'm Kelly...") → 捕获麦克风音频并完成中文转写 + turn-detection。
完整的 STT→LLM→TTS 语音闭环在本地跑通。

## 其它运行模式

- `uv run python agent.py dev` —— 连 LiveKit 服务器(需 `LIVEKIT_URL/API_KEY/API_SECRET`),
  可用浏览器 playground UI。
- `uv run python agent.py start` —— 生产模式。

## 关键代码

`agent.py` 的核心是把官方 `basic_agent.py` 的托管推理换成:

```python
session = AgentSession(
    llm=openai.realtime.RealtimeModel(voice="alloy"),  # 一个组件搞定 STT+LLM+TTS
)
```
