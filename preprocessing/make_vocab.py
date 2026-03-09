import json

with open("json/train_fbank.json", "r") as f:
    data = json.load(f)

phones = set()

for item in data.values():
    phns = item["phn"].split()
    for p in phns:
        phones.add(p)

phones = sorted(list(phones))
print("Number of unique phones:", len(phones))
print(phones)

        
with open("vocab_39.txt", "w") as f:
    f.write("_\n")  # blank token first
    for p in phones:
        f.write(p + "\n")

print("vocab_39.txt created.")