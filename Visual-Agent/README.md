# Visual-Agent · Trợ giảng STEM lớp 12

Visual-Agent là trợ giảng tự học đa phương thức cho Toán, Vật lí, Hóa học và Sinh học theo CTGDPT 2018. Học sinh có thể chụp đề, kiểm tra bản chép, nhận gợi ý Socrates theo từng mức và theo dõi năng lực theo chủ đề.

## Kiến trúc

- React/Vite: đăng nhập magic link, workspace học, hint ladder, nguồn, mastery và upload học liệu riêng.
- FastAPI/SSE: API xác thực, upload private và event học tập; không phát chain-of-thought.
- LangGraph 1.x: `perception → visualization decision/tool → retrieval → tutor → assessment → mastery`.
- Math-To-Manim tool: ưu tiên tạo animation cho câu hỏi khái niệm, công thức, định luật và hiện tượng Toán/Vật lí; scene được kiểm tra AST, render thật và tự sửa có giới hạn.
- Exam Research Agent: lấy thời gian hiện tại, phân tích câu đã duyệt của hai kỳ thi hoàn tất gần nhất và chuyển báo cáo có cấu trúc cho Learning Path Supervisor.
- Ollama Cloud trực tiếp qua `https://ollama.com/v1`: cấu hình riêng tutor, vision và research.
- Supabase: Auth/JWKS, PostgreSQL, pgvector, private Storage và RLS theo `auth.uid()`.
- Redis: checkpoint ngắn hạn và hàng đợi RQ; worker đồng bộ nguồn đề, scheduler quét hàng tuần và dọn upload hết hạn.

Research Agent và YOLO vẫn nằm trong codebase như công cụ mở rộng, nhưng không thuộc đường chạy mặc định của trợ giảng.

## Chạy local

Yêu cầu Python 3.11–3.12, Node.js 22, Ollama Cloud API key và Manim CE 0.19+; FFmpeg/LaTeX được khuyến nghị và đã có trong Docker image. Khi local chưa có LaTeX, code generator dùng `Text` cho công thức để vẫn render được. Redis/Supabase là tùy chọn cho local.

```powershell
Copy-Item .env.example .env
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

Ở terminal khác:

```powershell
Set-Location frontend
npm ci
npm run dev
```

Khi chưa cấu hình Supabase và `AUTH_REQUIRED=false`, API dùng một tài khoản dev cục bộ. Không dùng chế độ này trên môi trường public.

Khi gọi trực tiếp Ollama Cloud, dùng model tag `gemma4:31b`, `gpt-oss:120b` — không thêm hậu tố `-cloud`. Hậu tố đó chỉ dành cho đường chạy qua daemon Ollama local đã đăng nhập. Chat/vision tiếp tục dùng Cloud; embeddings dùng `nomic-embed-text` 768 chiều qua Ollama local tại `http://127.0.0.1:11434/v1`. Nếu daemon local tạm unavailable, ingestion vẫn lưu chunk và tự fallback sang lexical retrieval.

Visualization dùng `VISUALIZATION_MODEL` (mặc định `OLLAMA_MODEL=gemma4:31b-cloud`). Model factory tự bỏ hậu tố `-cloud` khi gọi trực tiếp `https://ollama.com/v1`; credential chỉ đọc từ `OLLAMA_API_KEY`. Artifact được lưu persistent dưới `VISUALIZATION_ARTIFACT_DIR`, cache theo `normalized_question + subject + model + VISUALIZATION_VERSION`, và retry tối đa `MAX_RENDER_RETRIES`.

## Thiết lập Supabase

1. Tạo project Supabase và chạy toàn bộ migration trong [supabase/migrations](supabase/migrations) theo thứ tự tên file.
2. Điền `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, đặt `AUTH_REQUIRED=true`.
3. Tạo `frontend/.env.local` với `VITE_SUPABASE_URL` và `VITE_SUPABASE_PUBLISHABLE_KEY`.
4. Thêm email giáo viên vào `ADMIN_EMAILS`.

Service-role key chỉ được dùng ở backend/worker. Bucket `uploads` là private, URL xem trước hết hạn sau 15 phút và job retention xóa tài sản sau 30 ngày.

## API chính

| Method | Endpoint | Chức năng |
|---|---|---|
| `GET` | `/api/me` | Danh tính và quyền admin |
| `POST` | `/api/uploads` | Upload ảnh thật ≤25 MB, trả `asset_id` |
| `POST` | `/api/knowledge` | Stage tài liệu riêng ≤25 MB, trả `202` và `job_id`; worker tự chunk/embedding ở nền |
| `GET` | `/api/knowledge/jobs` | Danh sách tiến trình ingestion thuộc user hiện tại |
| `GET` | `/api/knowledge/jobs/{job_id}` | Trạng thái một job thuộc user hiện tại |
| `GET` | `/api/knowledge/documents` | Danh mục tài liệu đã lưu cùng số chunks thực tế |
| `POST` | `/api/webhooks/knowledge-uploaded` | Webhook HMAC idempotent để kích hoạt job sau sự kiện upload |
| `GET` | `/api/chat/sessions` | Danh sách phiên chat persistent để mở lại trên UI |
| `POST` | `/api/tutor/stream` | SSE stream token realtime, stage, extraction, feedback, citation và mastery |
| `WS` | `/api/tutor/ws` | WebSocket JSON stream; gửi JWT và tutor request trong frame đầu tiên |
| `GET` | `/api/history/{session_id}` | Khôi phục problem, hint level và buffer hội thoại của phiên |
| `GET` | `/api/visualizations/{artifact_id}/video` | Stream MP4 thuộc user hiện tại |
| `GET` | `/api/visualizations/{artifact_id}/scene` | Tải source Manim thuộc user hiện tại |
| `GET` | `/api/me/mastery` | Hồ sơ năng lực 0–1 |
| `GET` | `/api/me/recommendations` | Chủ đề ưu tiên cá nhân hóa |
| `GET` | `/api/me/exam-plan/{subject}` | Phân bổ đề hai năm và lộ trình cá nhân hóa theo bốn thang điểm |
| `DELETE` | `/api/me` | Xóa tài sản private và dữ liệu học tập |
| `POST` | `/api/admin/content` | Adapter text-only cũ, deprecated; dùng `/api/knowledge` |
| `POST` | `/api/admin/exam-sync` | Đưa job đồng bộ nguồn đề vào RQ |
| `GET/PATCH` | `/api/admin/review-queue` | Duyệt dữ liệu OCR trước khi publish |

`POST /api/stream` là adapter deprecated trong một phiên bản để UI cũ tiếp tục nhận `agent_start`, `token`, `error`, `done`.

Tutor memory giữ tối đa 24 message cho mỗi key `user_id:session_id`. Khi có `REDIS_URL`, checkpoint tồn tại qua restart; nếu Redis chưa được cấu hình, backend dùng memory nội bộ và lịch sử mất khi process khởi động lại. Frontend ưu tiên WebSocket và fallback về SSE nếu chưa thiết lập được kết nối.

Lịch sử chat hiển thị được lưu riêng khỏi checkpoint: production dùng bảng Supabase `chat_messages` (migration `202608250001_persistent_chat_history.sql`), còn local development dùng SQLite tại `CHAT_DATABASE_PATH` (`data/chat_history.sqlite3` mặc định). Mỗi cặp user/assistant được upsert theo `user_id + session_id + request_id + role`, nên retry không tạo message trùng. Redis chỉ còn giữ state thực thi ngắn hạn của graph.

Kho knowledge cũng thuộc sở hữu user: API chỉ staging file và trả về ngay; event upload tự kích hoạt queue `knowledge-ingestion` (hoặc background fallback khi Redis không có). UI poll trạng thái bền vững nên người dùng có thể đóng panel và tiếp tục chat. Worker kiểm tra nội dung, trích text rồi chia đoạn tối đa 1.200 ký tự. Production lưu job/document/chunk và vector trong PostgreSQL/pgvector với RLS; local lưu bền vững trong cùng SQLite. Tutor chỉ truy xuất chunk của chính user và corpus global đã duyệt, không đọc học liệu riêng của tài khoản khác. Khi embedding endpoint chưa sẵn sàng, ingestion vẫn hoàn tất và retrieval dùng lexical fallback.

Webhook nhận JSON `{"job_id":"..."}` và yêu cầu header `X-Knowledge-Signature: sha256=<HMAC_SHA256(raw_body, KNOWLEDGE_WEBHOOK_SECRET)>`. Claim trạng thái `pending → processing` là atomic nên callback lặp lại không tạo document trùng.

Research Agent không dùng Wikipedia/arXiv để suy ra cấu trúc đề. Nó chỉ đọc `exam_questions` có `review_status=approved`, loại năm thi chưa hoàn tất theo `APP_TIMEZONE`, lấy tối đa hai năm gần nhất và ghi rõ coverage/nguồn. Supervisor kết hợp phân bổ 60% năm mới + 40% năm trước với mastery quan sát được để tạo lộ trình `0–5`, `5–7`, `7–8.5`, `8.5–10`; khi thiếu corpus đã duyệt, API trả `insufficient_data` thay vì sinh template.

## Worker và Docker

```powershell
docker compose up --build
```

Compose khởi động backend, frontend, Redis, RQ worker và RQ scheduler. Registry outbound chỉ cho phép các host Bộ GDĐT/Cổng Chính phủ trong `EXAM_SOURCE_HOSTS`; mọi câu OCR bắt đầu ở trạng thái `pending`.

## Kiểm thử

```powershell
python -m pytest -q
Set-Location frontend
npm audit --audit-level=high
npm run build
```

Các test hiện có bao phủ công thức mastery/recommendation, MIME/giải mã ảnh, SSRF allowlist, ingestion idempotent, validators STEM, nhánh clarification, API dev-auth, visualization routing, structured planner, AST gate, render repair, cache, ownership và text fallback. Các ngưỡng chất lượng trên golden set 100 ảnh và pilot học sinh cần dữ liệu/reviewer thật; chúng không được CI unit test tự tuyên bố là đã đạt.

## Luồng Math-To-Manim

Thiết kế artifact chain, AST gate và bounded render-repair được tích hợp theo kiến trúc của [HarleyCoops/Math-To-Manim](https://github.com/HarleyCoops/Math-To-Manim) (MIT), nhưng dùng trực tiếp Tutor Graph, model factory và API ownership của Visual-Agent thay vì nhúng các silo CLI/MCP của repository gốc.

```text
Câu hỏi
  → education/tutor_graph.py: visualization_router
  → tools/visualization/tool.py: visualize_stem_concept
  → tools/visualization/provider.py: Ollama Cloud / Gemma 4
  → tools/visualization/service.py: ConceptAnalysis + VisualizationPlan
  → tools/visualization/validation.py: syntax/AST security gate
  → tools/visualization/renderer.py: Manim subprocess
  → data/visualizations/<cache-key>/: scene, stage JSON, manifest, logs, MP4
  → api.py: WebSocket/SSE visualization event và protected artifact endpoint
  → frontend/src/App.jsx: video trong hội thoại
```

Gọi tool trực tiếp từ Python:

```python
result = await visualize_stem_concept(
    "Giải thích F = ma", subject="physics", level="THPT lớp 12", user_id=user_id
)
```

Nếu planning, validation hoặc render thất bại, tool trả `success=false`; Tutor Graph vẫn chạy retrieval/tutor và trả giải thích văn bản thay vì làm hỏng phiên chat.

## Cấu trúc mới

```text
education/              domain model, curriculum, Tutor Graph, mastery, validators
exam/                   source registry, fetch/OCR, jobs
services/               auth, repository, private storage
tools/visualization/     analyzer, planner, Manim codegen, validator, renderer, cache
supabase/migrations/    schema, pgvector, RLS và taxonomy seed
tests/                  test backend
frontend/src/           workspace học sinh và admin review
api.py                  FastAPI + SSE
worker.py               RQ worker
scheduler.py            lịch nguồn đề và retention
```
