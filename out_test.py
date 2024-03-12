
import json

lr = 5e-3
training_iters = 5
patch_size = 32
num_rotations = 16

params = {
    "lr": lr,
    "training_iters": training_iters,
    "patch_size": patch_size,
    "num_rotations": num_rotations
}

with open("./log/" + "params.json", "w") as f:
    json.dump(params, f)
