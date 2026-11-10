from runtime_paths import configured_path

path = configured_path("CARLA_ROOT") / "CarlaUE4" / "Config" / "DefaultEngine.ini"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("Town10HD_Opt.Town10HD_Opt", "Town03.Town03")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("DefaultEngine.ini updated to boot into Town03 by default!")
