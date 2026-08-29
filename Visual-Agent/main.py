from utils.inference import run_supervisor

if __name__ == "__main__":
    messages = run_supervisor(
        "Môn Toán: Em chưa biết bắt đầu bài tìm đạo hàm của y = x^3 - 2x từ đâu."
    )
    print(messages)
