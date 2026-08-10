from langchain_core.messages import convert_to_messages


def _pretty_print_message(message, indent=False):
    text = message.pretty_repr(html=False)
    if indent:
        text = "\n".join("\t" + line for line in text.split("\n"))
    print(text)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if not ns:
            return
        print(f"Update for graph: {ns[-1].split(':')[0]}\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        label = f"Update from node {node_name}:\n"
        if is_subgraph:
            label = "\t" + label
        print(label)
        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = [messages[-1]]
        for msg in messages:
            _pretty_print_message(msg, indent=is_subgraph)
        print()
