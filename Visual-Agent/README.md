# Visual-Agent · Trợ giảng STEM lớp 12

Visual-Agent là trợ giảng tự học đa phương thức cho Toán và Vật lí lớp 12 theo CTGDPT 2018. Học sinh có thể học sâu một khái niệm, chụp đề, kiểm tra bản chép, nhận gợi ý Socrates theo từng mức và theo dõi năng lực theo chủ đề.

## Kiến trúc

- React/Vite: đăng nhập magic link, workspace học, hint ladder, nguồn, mastery và upload học liệu riêng.
- FastAPI/SSE: API xác thực, upload private và event học tập; không phát chain-of-thought.
- LangGraph 1.x: `perception → visualization decision/tool → retrieval → supervisor → research → tutor → assessment → mastery`.
- Minh hoạ chat: model phối hợp đối tượng, quan hệ và biểu thức thành cảnh SVG 2D hoặc Three.js 3D, chạy song song với lời giải; xem [GENERAL-SCENES.md](GENERAL-SCENES.md). Manim là tool riêng cho video.
- Supervisor: đánh giá mức hiểu biết từ bằng chứng hội thoại/mastery của đúng môn và tạo hướng tiếp cận nền tảng → vận dụng → nâng cao.
- Research Agent: nhận lệnh từ Supervisor để tra cứu kiến thức, phương pháp và dạng bài cơ bản → trung bình → nâng cao làm ngữ cảnh cho Tutor.
- River DeepSeek qua `https://api.river.ai/v1` cho tutor và sinh dữ liệu minh hoạ; Ollama Cloud giữ Gemma cho mô tả ảnh, GPT-OSS cho research.
- Supabase: Auth/JWKS, PostgreSQL, pgvector, private Storage và RLS theo `auth.uid()`.
- Redis Stack: checkpoint LangGraph ngắn hạn, hàng đợi RQ và scheduler dọn upload hết hạn.

Research Agent nằm trong Tutor Graph cho cả yêu cầu tìm cách giải và hiểu sâu khái niệm; YOLO vẫn là công cụ thị giác tùy chọn và không thuộc đường chạy mặc định.

## Chạy local

Yêu cầu Python 3.11–3.12, Node.js 22, River API key, Ollama Cloud API key và Manim CE 0.19+; FFmpeg/LaTeX được khuyến nghị và đã có trong Docker image. Khi local chưa có LaTeX, code generator dùng `Text` cho công thức để vẫn render được. Khi chạy trực tiếp không qua Docker, PostgreSQL/Redis có thể bỏ qua để dùng fallback local. Docker development đã cấu hình sẵn PostgreSQL và Redis Stack.

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

`RIVER_BASE_URL=https://api.river.ai/v1` và `RIVER_API_KEY` dành cho hội thoại, đánh giá và minh hoạ. `LLM_MODEL_NAME` mặc định là `deepseek-ai/DeepSeek-V4.1-Flash`; `TUTOR_MODEL` và `VISUALIZATION_MODEL` có thể ghi đè tên model tại River. Thiếu River key sẽ báo lỗi cấu hình, không tự chuyển sang Gemma/GPT.

`VISION_MODEL=gemma4:31b` chỉ đọc/mô tả ảnh; `RESEARCH_MODEL=gpt-oss:120b` chỉ research. Hai vai trò này dùng `OLLAMA_BASE_URL` và `OLLAMA_API_KEY`. Model factory tự bỏ hậu tố `-cloud` khi gọi trực tiếp `https://ollama.com/v1`. Embeddings dùng `nomic-embed-text` 768 chiều qua endpoint `EMBEDDING_BASE_URL` riêng; nếu daemon local tạm unavailable, ingestion vẫn lưu chunk và tự fallback sang lexical retrieval. Minh hoạ chat dùng scene compiler trong [GENERAL-SCENES.md](GENERAL-SCENES.md); video Manim vẫn có artifact và giới hạn retry riêng.

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
| `POST` | `/api/chat/sessions` | Tạo phiên theo môn ở backend và lưu ngay vào database |
| `PATCH` | `/api/chat/sessions/{session_id}` | Đổi và lưu `learning_mode` của phiên |
| `POST` | `/api/tutor/stream` | SSE stream token realtime, stage, extraction, learning plan, feedback, citation và mastery |
| `WS` | `/api/tutor/ws` | WebSocket JSON stream; gửi JWT và tutor request trong frame đầu tiên |
| `GET` | `/api/history/{session_id}` | Khôi phục problem, hint level và buffer hội thoại của phiên |
| `GET` | `/api/visualizations/{artifact_id}/video` | Stream MP4 thuộc user hiện tại |
| `GET` | `/api/visualizations/{artifact_id}/scene` | Tải source Manim thuộc user hiện tại |
| `GET` | `/api/me/mastery` | Hồ sơ năng lực 0–1 |
| `GET` | `/api/me/recommendations` | Chủ đề ưu tiên cá nhân hóa |
| `DELETE` | `/api/me` | Xóa tài sản private và dữ liệu học tập |
| `POST` | `/api/admin/content` | Adapter text-only cũ, deprecated; dùng `/api/knowledge` |

`POST /api/stream` là adapter deprecated trong một phiên bản để UI cũ tiếp tục nhận `agent_start`, `token`, `error`, `done`.

Tutor memory giữ tối đa 24 message cho mỗi key `user_id:session_id`. Khi có `REDIS_URL`, checkpoint tồn tại qua restart; nếu Redis chưa được cấu hình, backend dùng memory nội bộ và lịch sử mất khi process khởi động lại. Frontend ưu tiên WebSocket và fallback về SSE nếu chưa thiết lập được kết nối.

Frontend không lưu session/message trong `localStorage`. Mỗi lần vào hoặc tải lại ứng dụng, UI tự tạo một phiên mới; mỗi môn giữ một phiên hoạt động riêng trong vòng đời trang nên chuyển qua lại Toán/Vật lí không trộn lịch sử hoặc tạo phiên lặp. Nút “Phiên mới” cũng tạo session mới trong PostgreSQL và reset workspace. Lịch sử cũ vẫn mở được qua `/api/chat/sessions?subject=...`. Mỗi session lưu `learning_mode=concept|guided_problem`; selector trên UI cập nhật trực tiếp session thay vì chèn prompt mẫu, và mode được khôi phục khi mở lịch sử. Backend khóa subject và từ chối dùng cùng một session cho môn khác. Các endpoint tạo phiên, upload học liệu và Tutor từ chối Hóa/Sinh nhưng enum cũ vẫn được giữ để đọc dữ liệu legacy mà không xóa dữ liệu người dùng.

Lịch sử chat hiển thị được lưu riêng khỏi checkpoint. Khi có `DATABASE_URL`, backend dùng các bảng `visual_agent.tutoring_sessions` và `visual_agent.chat_messages` trong PostgreSQL; Supabase/PostgREST vẫn được ưu tiên khi cấu hình đầy đủ. SQLite tại `CHAT_DATABASE_PATH` chỉ là fallback khi cả hai cấu hình database đều vắng mặt. Mỗi cặp user/assistant được upsert theo `user_id + session_id + request_id + role`, nên retry không tạo message trùng. Redis chỉ giữ state thực thi ngắn hạn của graph.

Kho knowledge cũng thuộc sở hữu user: API chỉ staging file và trả về ngay; event upload tự kích hoạt queue `knowledge-ingestion` (hoặc background fallback khi Redis không có). UI poll trạng thái bền vững nên người dùng có thể đóng panel và tiếp tục chat. Worker kiểm tra nội dung, trích text rồi chia đoạn tối đa 1.200 ký tự. Production lưu job/document/chunk và vector trong PostgreSQL/pgvector với RLS; local lưu bền vững trong cùng SQLite. Tutor chỉ truy xuất chunk của chính user và corpus global đã duyệt, không đọc học liệu riêng của tài khoản khác. Khi embedding endpoint chưa sẵn sàng, ingestion vẫn hoàn tất và retrieval dùng lexical fallback.

Webhook nhận JSON `{"job_id":"..."}` và yêu cầu header `X-Knowledge-Signature: sha256=<HMAC_SHA256(raw_body, KNOWLEDGE_WEBHOOK_SECRET)>`. Claim trạng thái `pending → processing` là atomic nên callback lặp lại không tạo document trùng.

Research Agent không fetch ngân hàng đề, đáp án, phổ điểm hoặc cấu trúc đề. `learning_mode` explicit trong session là nguồn quyết định chính; regex câu chữ chỉ là fallback cho client cũ không có mode. Supervisor gửi lệnh nghiên cứu có môn, khái niệm, mức hiểu biết và yêu cầu dạng bài phân cấp. Research Agent dùng hai system policy riêng: `guided_problem` tập trung phương pháp/bài tương tự cho hint ladder, còn `concept` tập trung trực giác/định nghĩa/ngộ nhận/bài luyện phân cấp. Agent dùng Tavily cho web, arXiv cho tài liệu học thuật và Wikipedia cho kiến thức nền; báo cáo giữ URL nguồn rồi được gửi làm ngữ cảnh cho Tutor. Cấu hình `TAVILY_API_KEY` trong `.env`. Tutor dùng Socrates cho bài cần giải và dùng luồng trực giác → định nghĩa → ví dụ → ngộ nhận → mở rộng cho yêu cầu học khái niệm.

Recommendation không còn phụ thuộc dữ liệu đề thi. Điểm ưu tiên hiện dùng `0,70 × khoảng trống mastery + 0,30 × tỷ lệ lỗi quan sát được` và chỉ sinh từ kỹ năng đã có bằng chứng thực tế.

## Worker và Docker

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f backend worker
```

Môi trường development mở frontend tại `http://localhost:5173` và backend tại `http://localhost:8000`. Compose khởi động PostgreSQL 17, Redis Stack có RediSearch/AOF, backend Uvicorn hot reload, frontend Vite HMR, RQ worker và scheduler. Source được bind mount nên sửa Python/React không cần build lại; volume riêng giữ PostgreSQL, Redis, uploads, artifacts và `node_modules`. Uvicorn loại `data`, `assets`, `uploads`, `frontend` và `tests` khỏi watcher để file Manim/OCR sinh trong runtime không restart backend giữa một WebSocket stream.

Backend Dockerfile có các target `builder`, `development`, `production` và `development-yolo`; frontend có `dependencies`, `development`, `builder`, `production`. YOLO/Ultralytics được tách khỏi image mặc định để tránh kéo Torch/CUDA vào luồng tutor. Chỉ build target tùy chọn khi thực sự cần object detection:

```powershell
docker build -f Dockerfile.backend --target development-yolo -t visual-agent-backend:yolo .
```

Dừng stack nhưng giữ database bằng `docker compose down`; chỉ dùng `docker compose down -v` khi chủ động muốn xóa toàn bộ dữ liệu development.

## Kiểm thử

```powershell
python -m pytest -q
Set-Location frontend
npm audit --audit-level=high
npm run build
```

Image development đã chứa pytest, vì vậy có thể test đúng môi trường container bằng `docker compose exec -T backend python -m pytest -q`.

Các test hiện có bao phủ công thức mastery/recommendation, routing và grounding của Research Agent, MIME/giải mã ảnh, ingestion idempotent, validators STEM, nhánh clarification, API dev-auth, visualization routing, structured planner, AST gate, render repair, cache, ownership và text fallback.

## Minh hoạ chat và tool video Manim

Chat mặc định dùng đề hiện hành, yêu cầu và ngữ cảnh để lập JSON cảnh tổng quát gồm objects, relations, parameters và biểu thức chuyển động. Bộ dựng hình kiểm tra dữ liệu rồi hiển thị SVG 2D hoặc Three.js 3D, không giới hạn vào bốn chủ đề có sẵn. Minh hoạ không chặn lời giải; kết quả về muộn được lưu vào cùng tin nhắn. Xem [GENERAL-SCENES.md](GENERAL-SCENES.md).

Luồng dưới đây chỉ dành cho tool video gọi trực tiếp, không phải luồng chat mặc định.

Thiết kế artifact chain, AST gate và bounded render-repair được tích hợp theo kiến trúc của [HarleyCoops/Math-To-Manim](https://github.com/HarleyCoops/Math-To-Manim) (MIT), nhưng dùng trực tiếp Tutor Graph, model factory và API ownership của Visual-Agent thay vì nhúng các silo CLI/MCP của repository gốc.

```text
Lời gọi tool video trực tiếp
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

Nếu planning, validation hoặc render thất bại, tool video trả `success=false`. Chat mặc định dùng mẫu cố định và báo rõ failed/unsupported khi không có minh hoạ phù hợp.

## Cấu trúc mới

```text
education/              domain model, curriculum, Tutor Graph, mastery, validators
services/               auth, repository, private storage
tools/visualization/     JSON templates, independent stream; standalone Manim tool
supabase/migrations/    schema, pgvector, RLS và taxonomy seed
tests/                  test backend
frontend/src/           workspace học sinh và admin review
api.py                  FastAPI + SSE
worker.py               RQ worker
scheduler.py            lịch dọn upload hết hạn
```
