"""Multiplex independent illustration work with graph output, without graph barriers."""
import asyncio

from tools.visualization.scene_service import SceneIllustrationService


async def illustration_stream(tutor, graph_input, config, *, service=None):
    queue = asyncio.Queue()
    tasks = []
    finished = object()
    service = service or SceneIllustrationService()

    async def visualize(decision):
        try:
            result = await service.visualize(
                question=decision["question"], subject=decision["subject"],
                user_id=graph_input["user_id"],
                progress=lambda stage: queue.put_nowait(("custom", {"type": "visualization_progress", "stage": stage})),
            )
            await queue.put(("updates", {"visualization_tool": {"visualization": result.model_dump(mode="json")}}))
        except Exception:
            await queue.put(("updates", {"visualization_tool": {"visualization": {
                "success": False, "status": "failed", "error": "Không thể tạo minh hoạ trong lượt này."
            }}}))

    async def graph():
        try:
            async for raw in tutor.astream(graph_input, config=config, stream_mode=["custom", "updates"]):
                await queue.put(raw)
                if isinstance(raw, tuple) and raw[0] == "updates":
                    decision = raw[1].get("visualization_router", {}).get("visualization_decision", {})
                    if decision.get("should_visualize") and decision.get("question") and not tasks:
                        tasks.append(asyncio.create_task(visualize(decision)))
        except Exception as exc:
            await queue.put(exc)
        finally:
            await queue.put(finished)

    graph_task = asyncio.create_task(graph())
    try:
        graph_done = False
        while not graph_done or any(not task.done() for task in tasks) or not queue.empty():
            # Once graph is done, wait for the independent illustration to enqueue
            # its result; it always has a bounded model deadline.
            if graph_done and queue.empty() and tasks:
                await asyncio.gather(*tasks)
                if queue.empty():
                    break
            item = await queue.get()
            if item is finished:
                graph_done = True
            elif isinstance(item, Exception):
                raise item
            else:
                yield item
    finally:
        for task in [graph_task, *tasks]:
            if not task.done():
                task.cancel()
        await asyncio.gather(graph_task, *tasks, return_exceptions=True)
