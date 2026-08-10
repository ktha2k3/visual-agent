# Hệ Thống Visual Agent

Đây là một hệ thống AI đa tác tử được xây dựng trên [LangGraph](https://github.com/langchain-ai/langgraph), kết hợp **Research Agent** (arXiv + Wikipedia) và **Vision Agent** (mô tả ảnh + phát hiện đối tượng bằng YOLO), được điều phối bởi một **Supervisor** để định tuyến tác vụ và tổng hợp câu trả lời. Dự án đi kèm giao diện React hỗ trợ streaming.

---

## Kiến Trúc Hệ Thống

```
Người dùng (Frontend)
      │
      ▼  HTTP POST /api/stream  (Server-Sent Events)
┌─────────────────────────────────┐
│         FastAPI  (api.py)       │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       Supervisor Agent          │  ← kimi-k2.5:cloud qua Ollama
│ (đồ thị Supervisor của LangGraph)│
└────────┬────────────────┬───────┘
         │                │
         ▼                ▼
┌────────────────┐  ┌─────────────────┐
│ Research Agent │  │  Vision Agent   │  ← được gọi khi cần
│ gpt-oss:120b   │  │  gemma4:31b     │
└───────┬────────┘  └────────┬────────┘
        │                    │
   ┌────┴────┐         ┌─────┴───────────┐
   │ arxiv   │         │ describe_image  │
   │wikipedia│         │ detect (YOLOv11x)│
   └─────────┘         └─────────────────┘
```

### Vai trò của các agent

| Agent | Model | Trách nhiệm |
|---|---|---|
| **Supervisor** | kimi-k2.5:cloud | Định tuyến yêu cầu, tổng hợp câu trả lời cuối cùng |
| **Research Agent** | gpt-oss:120b-cloud | Tìm kiếm bài báo trên arXiv và bài viết trên Wikipedia |
| **Vision Agent** | gemma4:31b-cloud | Mô tả ảnh, phát hiện và đếm đối tượng (YOLOv11x) |

Tất cả model được phục vụ cục bộ qua **Ollama** tại `http://localhost:11434`.

---

## Cấu Trúc Dự Án

```
Visual-Agent/
├── agents/                    # Chỉ chứa định nghĩa agent
│   ├── supervisor_agent.py    # Supervisor — định tuyến và tổng hợp
│   ├── research_agent.py      # Research agent
│   └── vision_agent.py        # Vision agent
│
├── tools/                     # Cài đặt tool (mỗi file một trách nhiệm)
│   ├── research/
│   │   └── search_tools.py    # Tool arxiv + wikipedia
│   └── vision/
│       ├── describe_tool.py   # Mô tả ảnh qua multimodal LLM
│       └── detect_tool.py     # Phát hiện đối tượng qua YOLOv11x
│
├── utils/                     # Tiện ích dùng chung
│   ├── pretty_print.py        # Format debug output
│   └── inference.py           # Hỗ trợ chạy từ CLI
│
├── models/
│   └── yolo11x.pt             # Trọng số YOLOv11x (được track bằng Git LFS)
│
├── assets/                    # Ảnh mẫu để test
│
├── frontend/                  # Giao diện React + Vite
│   ├── src/
│   │   ├── App.jsx            # Panel hoạt động agent + panel phản hồi Supervisor
│   │   └── App.css
│   ├── nginx.conf             # Cấu hình reverse-proxy cho Docker
│   └── Dockerfile
│
├── api.py                     # Server FastAPI với SSE streaming
├── main.py                    # Điểm vào CLI
├── Dockerfile.backend
├── docker-compose.yml
└── requirements.txt
```

---

## Yêu Cầu Trước Khi Chạy

| Thành phần | Phiên bản |
|---|---|
| Python | 3.11 + |
| Node.js | 18 + |
| [Ollama](https://ollama.com) | mới nhất |
| Docker + Docker Compose | tùy chọn |

### Tải các model Ollama cần thiết

```bash
ollama signin
ollama pull kimi-k2.5:cloud
ollama pull gpt-oss:120b-cloud
ollama pull gemma4:31b-cloud
```

---

## Thiết Lập Môi Trường

Tạo file `.env` ở thư mục gốc của dự án (file này đã được git ignore):

```env
# Ollama
OLLAMA_API_KEY=ollama
# Ghi đè khi chạy trong Docker (container không truy cập được localhost)
# OLLAMA_BASE_URL=http://host.docker.internal:11434/v1

# Tùy chọn — tìm kiếm web bằng Tavily
TAVILY_API_KEY=your_tavily_key

# Tùy chọn — trace bằng LangSmith
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=visual-agent

# Tùy chọn — ghi đè đường dẫn model YOLO (mặc định: models/yolo11x.pt)
# YOLO_MODEL_PATH=models/yolo11x.pt
```

---

## Chạy Hệ Thống

### Cách A — Phát triển cục bộ (khuyến nghị khi đang chỉnh sửa)

**1. Cài dependency Python**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Khởi động backend**

```bash
uvicorn api:app --reload --port 8000
```

**3. Khởi động frontend**

```bash
cd frontend
npm install
npm run dev
```

Mở [http://localhost:5173](http://localhost:5173).

---

### Cách B — Docker compose (Recommend for test)

```bash
docker compose up --build -d
cd frontend
npm install
npm run dev
```

Mở [http://localhost:5173](http://localhost:5173).

---

## Tài Liệu API

### `POST /api/stream`

Stream hoạt động của agent dưới dạng [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events).

**Request body**

```json
{ "message": "Có bao nhiêu người trong ảnh assets/image.png?" }
```

**Các loại event**

| `type` | Trường dữ liệu | Mô tả |
|---|---|---|
| `agent_start` | `agent` | Một agent/node bắt đầu hoạt động |
| `token` | `agent`, `content` | Token văn bản được stream ra |
| `tool_call` | `agent`, `tool`, `args` | Agent đang gọi một tool |
| `tool_result` | `agent`, `tool`, `content` | Tool đã trả kết quả |
| `handoff` | `agent`, `tool` | Điều hướng nội bộ giữa các agent (UI bỏ qua) |
| `error` | `content` | Có exception xảy ra |
| `done` | `supervisor_responded` | Stream hoàn tất |

**Test nhanh bằng curl**

```bash
curl -X POST http://localhost:8000/api/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Các bài báo mới nhất về kiến trúc transformer"}' \
  --no-buffer
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## Luồng Xử Lý Một Request

```
1. Người dùng nhập tin nhắn trên UI
2. Frontend gửi POST /api/stream
3. Supervisor LLM quyết định gọi agent nào
4. Sub-agent chạy (research hoặc vision) và gọi các tool tương ứng
5. Sub-agent chuyển kết quả lại cho supervisor
6. Supervisor tổng hợp câu trả lời cuối cùng
7. Mọi bước đều được stream dưới dạng SSE và render theo thời gian thực trên UI
```

**Bố cục UI:**

```
┌──────────────────────┬─────────────────────────┐
│   Hoạt động Agent    │  Phản hồi Supervisor    │
│                      │                         │
│  🧠 Suy luận Agent   │  Câu trả lời cuối cùng  │
│    └ tool call       │  được render dưới dạng  │
│    └ tool result     │  Markdown               │
│  🔧 Thực thi Tool    │                         │
│  🔬 Research Agent   │                         │
└──────────────────────┴─────────────────────────┘
│  Nhập tin nhắn của bạn…               [Gửi]    │
└─────────────────────────────────────────────────┘
```

---

## Hướng Dẫn Phát Triển

### Thêm một tool mới

1. Tạo file trong `tools/research/` hoặc `tools/vision/`.
2. Gắn decorator `@tool` từ `langchain_core.tools` cho hàm của bạn.
3. Thêm tool đó vào danh sách `tools=[...]` của agent tương ứng trong `agents/`.

### Thêm một agent mới

1. Định nghĩa agent trong `agents/your_agent.py` bằng `create_react_agent`.
2. Import và đăng ký agent đó trong `agents/supervisor_agent.py` tại `agents=[...]`.
3. Thêm một mục vào `AGENT_META` trong `frontend/src/App.jsx` để khai báo icon và màu hiển thị trên UI.

### Đổi model

Thay chuỗi `model=` trong file agent tương ứng rồi tải model mới:

```bash
ollama pull <ten-model-moi>
```

### Bật LangSmith tracing

Đặt `LANGSMITH_TRACING=true` và `LANGSMITH_API_KEY` trong `.env` để lấy đầy đủ
trace thực thi LangGraph tại [smith.langchain.com](https://smith.langchain.com).

---

## Demo

[assets/demo_test.mp4](assets/demo_test.mp4)


