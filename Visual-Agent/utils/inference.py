from agents.supervisor_agent import supervisor


def get_final_messages(last_chunk):
    if last_chunk is None:
        return []
    if isinstance(last_chunk, tuple):
        _, last_chunk = last_chunk
    if isinstance(last_chunk, dict) and "supervisor" in last_chunk:
        return last_chunk["supervisor"].get("messages", [])
    if isinstance(last_chunk, dict):
        for node_update in last_chunk.values():
            if isinstance(node_update, dict) and "messages" in node_update:
                return node_update["messages"]
    return []


def run_supervisor(user_input: str, recursion_limit: int = 50):
    config = {"recursion_limit": recursion_limit}
    last_chunk = None
    for chunk in supervisor.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    ):
        last_chunk = chunk
    return get_final_messages(last_chunk)
