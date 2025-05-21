

class ADScenarios:
    def __init__(self, world, blueprint_library, ambush_path, attack_behavior = ("crossing", "borrow_lane", "merge")):
        self.world = world                          # carla world
        self.blueprint_library = blueprint_library  # carla blueprint library
        self.ambush_path = ambush_path              # ambush path
        self.attack_type = "crossing"

        self.actor_list = []

    def attack_behavior(self):

        # where ?
        # where start the attack behavior

        # when ?
        # when to trigger the attack behavior

        # how ?
        # velocity and its direction
        pass

    def __del__(self):
        for actor in self.actor_list:
            actor.destroy()

        print(f"All attackers have been killed !")