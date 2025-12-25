from pathlib import Path

def main():
    Path("results").mkdir(exist_ok=True)
    (Path("results") / "hello.txt").write_text("main.py ran successfully\n")
    print("OK - wrote results/hello.txt")

if __name__ == "__main__":
    main()
