"""Default chat illustration service: compose a scene, never generate code."""
import asyncio
import logging
import json
import hashlib
import os
import tempfile
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from tools.visualization.models import StemVisualizationResult
from tools.visualization.provider import OllamaProvider
from tools.visualization.scene import SCENE_PROMPT, SceneSpec, compile_scene
from tools.visualization.sampling import SamplingLimitError
from utils.config import get_settings


class SceneIllustrationService:
    def __init__(self, provider=None, timeout=20, diagnostic_dir=None):
        self.provider = provider or OllamaProvider()
        self.timeout = timeout
        self.diagnostic_dir = Path(diagnostic_dir) if diagnostic_dir is not None else Path(get_settings().visualization_artifact_dir)/"scene-errors"

    def _record_failure(self, run_id, attempt, user_id, question, spec, raw, error):
        """Private local diagnostics only: no public download endpoint or credentials."""
        temporary=None
        try:
            self.diagnostic_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
            record={"diagnostic_id":run_id,"attempt":attempt,
                    "created_at":datetime.now(timezone.utc).isoformat(),
                    "owner_hash":hashlib.sha256(user_id.encode()).hexdigest(),
                    "question":question[:18000],"scene":spec.model_dump(mode="json") if spec else None,
                    "raw":str(raw)[:64000] if raw is not None else None,
                    "error_type":type(error).__name__,"error":str(error)[:2000]}
            fd,temporary=tempfile.mkstemp(dir=self.diagnostic_dir,suffix=".tmp")
            with os.fdopen(fd,"w",encoding="utf-8") as handle:
                json.dump(record,handle,ensure_ascii=False,allow_nan=False)
            os.replace(temporary,self.diagnostic_dir/f"{run_id}-{attempt}.json")
        except Exception as exc:
            logging.getLogger(__name__).warning("Cannot write scene diagnostic (%s)",type(exc).__name__)
        finally:
            try:
                if temporary and os.path.exists(temporary):os.unlink(temporary)
            except OSError:
                logging.getLogger(__name__).warning("Cannot remove temporary scene diagnostic")

    async def visualize(self, *, question, subject, user_id, level=None, progress=None):
        base = dict(subject=subject, concept="Minh hoạ", explanation="", key_takeaway="", attempts=1)
        run_id=uuid.uuid4().hex
        deadline=time.monotonic()+self.timeout
        spec,raw=None,None
        try:
            if progress:
                progress("scene_composition")
            async with asyncio.timeout(self.timeout):
                prompt = f"Môn: {subject}. Trình độ: {level or 'THPT'}.\n{question[:18000]}"
                try:
                    spec = await self.provider.structured(SceneSpec, system=SCENE_PROMPT, prompt=prompt)
                    scene = await asyncio.to_thread(compile_scene, spec, deadline=deadline) if spec.objects else None
                except (ValueError, SyntaxError, ArithmeticError, KeyError) as exc:
                    if isinstance(exc,SamplingLimitError):
                        raise
                    if not hasattr(self.provider, "text"):
                        raise
                    invalid = spec.model_dump_json() if spec else ""
                    if isinstance(exc, ValidationError):
                        invalid = str(exc.errors()[0].get("input", ""))[:24000]
                    self._record_failure(run_id,base["attempts"],user_id,question,spec,invalid,exc)
                    base["diagnostic_id"]=run_id
                    if progress:
                        progress("scene_validation")
                    # One data-only repair, within the same total deadline. No render retry.
                    base["attempts"] = 2
                    spec=None
                    raw = await self.provider.text(system=SCENE_PROMPT + "\nSửa dữ liệu JSON dựa trên lỗi, không đổi nội dung đề.",
                        prompt=prompt + "\nLỖI KIỂM TRA:\n" + str(exc)[:1800] + "\nDỮ LIỆU CẦN SỬA:\n" + invalid[:24000]
                        + "\nJSON Schema:\n" + json.dumps(SceneSpec.model_json_schema(), ensure_ascii=False))
                    spec = SceneSpec.model_validate_json(OllamaProvider._extract_json_object(raw))
                    scene = await asyncio.to_thread(compile_scene, spec, deadline=deadline) if spec.objects else None
                if not spec.objects:
                    return StemVisualizationResult(**base, success=False, status="unsupported",
                        error=spec.reason or "Cần bổ sung dữ kiện hoặc tài liệu hình ảnh để minh hoạ yêu cầu này.")
            base["concept"] = spec.title
            if progress:
                progress("completed")
            return StemVisualizationResult(**base, success=True, status="completed", scene_data=scene)
        except Exception as exc:
            self._record_failure(run_id,base["attempts"],user_id,question,spec,raw,exc)
            base["diagnostic_id"]=run_id
            logging.getLogger(__name__).warning("Scene illustration failed diagnostic=%s attempt=%s (%s)",run_id,base["attempts"],type(exc).__name__)
            return StemVisualizationResult(**base, success=False, status="failed",
                error="Miền vẽ hoặc thời gian mô phỏng quá rộng để thể hiện chính xác trong giới hạn hiện tại. Hãy thu hẹp miền hoặc thời gian mô phỏng."
                if isinstance(exc,SamplingLimitError) else "Chưa dựng được cảnh hợp lệ cho yêu cầu này. Lời giải văn bản vẫn được giữ lại.")
