# Open and read your file
with open('numbered_output.txt', 'r', encoding='utf-8') as f:
    numbered_lines = f.readlines()

# Convert to a dictionary for easy lookup
line_dict = {}
for line in numbered_lines:
    if line.startswith('('):
        idx = int(line.split(')')[0][1:])
        line_dict[idx] = line.strip()

def parse_input(input_str):
    result = []
    parts = input_str.split(',')
    for part in parts:
        if '-' in part:
            start, end = part.split('-')
            result.extend(range(int(start.strip()), int(end.strip()) + 1))
        else:
            if part.strip().isdigit():
                result.append(int(part.strip()))
    return result

while True:
    input1 = input("Enter the first list of numbers (comma-separated, ranges allowed with '-', or type 'exit' to quit): ")
    if input1.lower() == 'exit':
        break
    input2 = input("Enter the second list of numbers (comma-separated, ranges allowed with '-', or type 'exit' to quit): ")
    if input2.lower() == 'exit':
        break

    list1 = parse_input(input1)
    list2 = parse_input(input2)

    print("\nFirst List Output:")
    for num in list1:
        if num in line_dict:
            print(line_dict[num])
        else:
            print(f"(Missing entry for number {num})")

    print("\n" + "_"*50 + "\n")

    print("Second List Output:")
    for num in list2:
        if num in line_dict:
            print(line_dict[num])
        else:
            print(f"(Missing entry for number {num})")
    print("\n---\n")