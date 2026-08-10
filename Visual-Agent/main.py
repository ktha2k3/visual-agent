from utils.inference import run_supervisor

if __name__ == "__main__":
    messages = run_supervisor(
        "What is this image about? Find related information: assets/image.png"
    )
    print(messages)
