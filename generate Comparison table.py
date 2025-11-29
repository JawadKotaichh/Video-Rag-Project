import pandas as pd

def main():
    data = {
        "Method": ["FAISS-Text", "TF-IDF", "BM25"],
        "Accuracy (%)": [46.7, 20.0, 13.3],
        "Latency (ms)": [11.8, 0.7, 0.8],
    }

    df = pd.DataFrame(data)

    # Print as a Markdown table
    print("\n## Retrieval Method Comparison\n")
    print(df.to_markdown(index=False))

    # Save to CSV for further use (e.g., in VS Code)
    df.to_csv("comparison_table.csv", index=False)

    # Also write a standalone Markdown file
    with open("comparison_table.md", "w") as f:
        f.write("## Retrieval Method Comparison\n\n")
        f.write(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
