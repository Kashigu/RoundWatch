from runtime_paths import configured_path

path = configured_path("CARLA_ROOT") / "CarlaUE4" / "Config" / "DefaultEngine.ini"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("Town03.Town03", "Town10HD_Opt.Town10HD_Opt")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Reverted DefaultEngine.ini to original Town10HD_Opt!")
