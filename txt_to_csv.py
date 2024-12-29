import csv

def convert_txt_to_csv(input_path, output_path):
    """
    將指定的 TXT 檔案轉換為 CSV 檔案。

    Args:
        input_path (str): 要讀取的 TXT 檔案路徑。
        output_path (str): 要寫入的 CSV 檔案路徑。
    """
    # 讀取 TXT 檔案
    with open(input_path, 'r', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()

    # 打開 CSV 檔案進行寫入
    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # 將每行寫入 CSV 檔
        for line in lines:
            # 假設每行是用空格分隔的，可以根據實際情況修改分隔符
            row = line.strip().split()
            writer.writerow(row)

    print(f"轉換完成！輸出位置: {output_path}")

input_file = 'O6310.txt'  # 更改為你的輸入 G-code 檔案位置
output_file = 'O6310.csv'  # 更改為你的輸出 CSV 檔案位置
convert_txt_to_csv(input_file, output_file)
