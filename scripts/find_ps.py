import psutil

print("Looking for processes...")
for p in psutil.process_iter(['pid', 'name', 'exe']):
    try:
        name = p.info['name'].lower()
        if 'carla' in name or 'ue4' in name or 'unreal' in name:
            print(p.info)
    except:
        pass
print("Done scan.")

