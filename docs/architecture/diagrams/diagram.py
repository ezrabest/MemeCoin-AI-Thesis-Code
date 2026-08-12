import os

# Set your target directory path
folder_path = r"E:\Projects\Final Project\memecoin_trader\docs\architecture\diagrams"

# Verify the path exists
if not os.path.exists(folder_path):
    print(f"Error: The path '{folder_path}' does not exist.")
else:
    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".mmd"):
            file_path = os.path.join(folder_path, filename)
            print("=" * 50)
            print(f"FILE: {filename}")
            print("=" * 50)

            # Read and print the content
            with open(file_path, "r", encoding="utf-8") as file:
                print(file.read())
                print("\n")
