def label_printer(predicted: str, true: str, filename: str):
    with open(f"output/{filename}.txt", "w", encoding="utf-8") as f:
        f.write("predicted:\n")
        f.write(predicted)
        f.write("\n")
        f.write("true:\n")
        f.write(true)
