from pathlib import Path

train_good = len(list(Path("data/casting_data/casting_data/train/ok_front").glob("*")))
train_defective = len(list(Path("data/casting_data/casting_data/train/def_front").glob("*")))

test_good = len(list(Path("data/casting_data/casting_data/test/ok_front").glob("*")))
test_defective = len(list(Path("data/casting_data/casting_data/test/def_front").glob("*")))

print("Train Good:", train_good)
print("Train Defective:", train_defective)

print("Test Good:", test_good)
print("Test Defective:", test_defective)