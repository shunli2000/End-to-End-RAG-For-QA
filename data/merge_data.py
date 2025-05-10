import json

def merge_files():
    files_to_merge = [
        "data/test/question_Zijin.txt",
        "data/test/questions_shun.txt",
        "data/test/test_questions_Youyou.txt"
    ]
    with open("data/test/questions.txt", "w", encoding='utf-8') as outfile:
        for fname in files_to_merge:
            try:
                with open(fname, encoding='utf-8') as infile:
                    outfile.write(infile.read())
            except UnicodeDecodeError:
                with open(fname, encoding='latin-1') as infile:
                    outfile.write(infile.read())
            outfile.write("\n")

def merge_reference_answers():
    files_to_merge = [
        "data/test/reference_answers_Zijin.json",
        "data/test/test_answers_shun.json",
        "data/test/test_reference_answers_Youyou.txt"
    ]
    merged_data = {}
    current_index = 1

    for fname in files_to_merge:
        with open(fname, encoding='utf-8') as infile:
            data = json.load(infile)
            for key in sorted(data.keys(), key=lambda x: int(x)):
                merged_data[str(current_index)] = data[key]
                current_index += 1

    with open("data/test/reference_answers.json", "w", encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, indent=4)

if __name__ == "__main__":
    merge_files()
    merge_reference_answers()
