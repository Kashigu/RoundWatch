path = r"E:\Simulation_Engines\CARLA_0.9.15\CarlaUE4\Config\DefaultEngine.ini"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("Town10HD_Opt.Town10HD_Opt", "Town03.Town03")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("DefaultEngine.ini updated to boot into Town03 by default!")

