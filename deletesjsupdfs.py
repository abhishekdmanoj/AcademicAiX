import os

folder = r"C:\study\projects\Main_project\AcademicAiX\data\raw_pdfs"

deleted = 0

for filename in os.listdir(folder):
    if filename.startswith("San_Jose_State_University"):
        file_path = os.path.join(folder, filename)
        try:
            os.remove(file_path)
            print(f"Deleted: {filename}")
            deleted += 1
        except Exception as e:
            print(f"Failed to delete {filename}: {e}")

print(f"\nTotal deleted: {deleted}")