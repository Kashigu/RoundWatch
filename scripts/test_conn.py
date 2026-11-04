import carla

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)
world = client.get_world()
print("SUCCESS! Connected to map:", world.get_map().name)

