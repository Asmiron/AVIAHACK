import os
import sys
import test


def find_stop_file(current_dir, stop_file_name):
    current_level = 0
    while current_level <= 2:
        for file in os.listdir(current_dir):
            if file == stop_file_name:
                return os.path.join(current_dir, file)
        current_dir = os.path.dirname(current_dir)
        current_level += 1
    return None


def has_numeric_directories(folder):
    directories = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    for directory in directories:
        if directory.isdigit() or (
                    directory.startswith('0.') and all(c.isdigit() or c == '.' for c in directory[2:])):
            return True
    return False


def is_numeric_folder(folder_name):
    return folder_name.isdigit() or (
                folder_name.startswith('0.') and all(c.isdigit() or c == '.' for c in folder_name[2:]))


def traverse_folders(folder, stop_file_name, stop_file_path=None):
    numeric_folders = []

    for root, dirs, files in os.walk(folder):
        if not stop_file_path:
            stop_file_path = find_stop_file(root, stop_file_name)
            if stop_file_path:
                print(f"Найден файл '{stop_file_name}' на уровне: {stop_file_path}")
                dirs.clear()

        for directory in dirs:
            if is_numeric_folder(directory):
                numeric_folders.append(directory)
                subdir = os.path.join(root, directory)
                _, stop_file_path = traverse_folders(subdir, stop_file_name, stop_file_path=stop_file_path)

    return numeric_folders, stop_file_path


def start_and_go(directory, name):
    folder_to_traverse = directory
    if not os.path.isdir(folder_to_traverse):
        print("Указанный путь не является папкой.")
        return

    stop_file_name = 'p'
    numeric_folders, stop_file_path = traverse_folders(folder_to_traverse, stop_file_name)

    if numeric_folders:
        numeric_values = [float(folder) if '.' in folder else int(folder) for folder in numeric_folders]
        max_value = max(numeric_values)
        print(f"Максимальное числовое значение среди папок: {max_value}")
        test.init(directory, max_value, name)
    else:
        print("В указанной папке нет числовых папок.")



def recursive_traverse_folders(folder):
    directories = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    name = ""
    if folder.__contains__('agard'):
        name = 'agard'
    elif folder.__contains__('data_wage') or folder.__contains__('data_step'):
        name = 'obstacle'
    if has_numeric_directories(folder):
        start_and_go(folder, name)

        return

    for directory in directories:
        recursive_traverse_folders(directory)


def main():
    name=""
    folder = sys.argv[1]
    directories = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

    for directory in directories:
        recursive_traverse_folders(directory)
    if has_numeric_directories(folder):
        if folder.__contains__('agard'):
            name = 'agard'
        elif folder.__contains__('data_wage') or folder.__contains__('data_step'):
            name = 'obstacle'
        start_and_go(folder, name)



if __name__ == "__main__":
    main()
