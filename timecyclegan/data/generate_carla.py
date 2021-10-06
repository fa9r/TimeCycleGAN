"""
Script to generate video sequences with semantic segmentation maps in Carla 0.9.6 (http://carla.org/).
The initial implementation of this script was written by Chenguang Huang.
Usage:  1. start the Carla server with: `./CarlaUE4.sh -carla-server -benchmark -fps=20 -quality-level=Low`
        2. run this client script from CARLA_0.9.6/PythonAPI/examples
"""

import argparse
import datetime
import glob
import os
import queue
import random
import sys
import time

import numpy as np
from PIL import Image
import pygame

from colormaps import create_carla_label_colormap, create_carla_label_colormap_cityscapes_style
from timecyclegan.util.os_utils import make_dir

try:
    sys.path.append(glob.glob('../carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')[0])
except IndexError:
    pass
import carla


def semantic_image_generator(raw_data, output_path, width, height):
    """
    convert raw semantic data to rgb image with different colors representing different objects
    :param raw_data: array of size W X H, each element stores an integer corresponding to a certain class of object
    :return: W X H X 3 np.array type
    """
    raw_data = np.frombuffer(raw_data, dtype=np.uint8)
    raw_data = raw_data.reshape(height, width, -1)[:, :, 2:3]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    color_map = create_carla_label_colormap_cityscapes_style()
    for i in range(height):
        for j in range(width):
            output[i, j, :] = color_map[int(raw_data[i, j])]
    output = Image.fromarray(output)
    output.save(output_path)
    return output


def run_carla(
        fps=20, time_inter=1, num_frames=200, other_vehicles=80, pedestrians=0, window_height=256, window_width=512,
        fov=120, sensor_tick=0.0, output_dir="./images/", town='Town04'
):
    """
    Run Carla and record RGB images and semantic segmentation maps
    :param fps: At how many FPS the simulation should run
    :param time_inter: Time interval between recorded frames
    :param num_frames: How many images/semseg-maps to record
    :param other_vehicles: Number of other vehicles in the simulation
    :param pedestrians: Number of pedestrians in the simulation
    :param window_height: Height of the Carla window
    :param window_width: Width of the Carla window
    :param fov: Horizontal field of view
    :param sensor_tick: Number of seconds between sensor measurements
    :param output_dir: Directory where data will be saved
    :param town: Name of the town to simulate
    """

    # ----- PREPARATION ------------------------------------------------------------------------------------------------
    print("\n##### INITIALIZING SIMULATION #####")

    # after how many milliseconds to stop
    stop_after = num_frames * time_inter

    # create folders for storing data
    semseg_dir = make_dir(os.path.join(output_dir, "semseg"))
    frame_dir = make_dir(os.path.join(output_dir, "frames"))

    # initialize actor_list, define Carla client, load world
    actor_list = []
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(2.0)

    # load world; this randomly throws a stupid timeout error sometime, so we have to wrap it like this...
    print("Trying to load world...")
    while True:
        try:
            # ----- CREATE WORLD AND ENABLE SYNCHRONOUS MODE -----------------------------------------------------------
            world = client.load_world(town)
            print('Enabling synchronous mode...')
            settings = world.get_settings()
            settings.fixed_delta_seconds = 1 / fps
            settings.synchronous_mode = True
            world.apply_settings(settings)
            print(town, "loaded successfully!")
            break
        except RuntimeError:
            print("Loading world", town, "failed. Retrying...")
            time.sleep(3)


    try:
        # ----- BLUEPRINTS ---------------------------------------------------------------------------------------------
        blueprints = world.get_blueprint_library()
        vehicle_blueprint = blueprints.filter("vehicle.*")
        pedestrian_blueprint = blueprints.filter("walker.*")
        pedestrian_controller_blueprint = world.get_blueprint_library().find('controller.ai.walker')

        # semseg blueprint
        sensor_blueprint = blueprints.find('sensor.camera.semantic_segmentation')
        sensor_blueprint.set_attribute('image_size_x', str(window_width))
        sensor_blueprint.set_attribute('image_size_y', str(window_height))
        sensor_blueprint.set_attribute('fov', str(fov))
        sensor_blueprint.set_attribute('sensor_tick', str(sensor_tick))

        # RGB blueprint
        rgb_blueprint = blueprints.find('sensor.camera.rgb')
        rgb_blueprint.set_attribute('image_size_x', str(window_width))
        rgb_blueprint.set_attribute('image_size_y', str(window_height))
        rgb_blueprint.set_attribute('fov', str(fov))
        rgb_blueprint.set_attribute('sensor_tick', str(sensor_tick))

        # ----- SPAWN POINTS -------------------------------------------------------------------------------------------
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # test whether vehicle number exceeds spawn points number
        if len(spawn_points) - 1 >= other_vehicles:  # - 1 for own car
            veh_num = other_vehicles
            rest = len(spawn_points) - other_vehicles - 1  # - 1 for own car
            if rest >= pedestrians:
                ped_num = pedestrians
            else:
                ped_num = rest
                print("Do not have so many spawn points. %d vehicles and %d pedestrains created." % (veh_num, ped_num))
        else:
            veh_num = len(spawn_points) - 1  # -1 for own car
            ped_num = 0
            print("Do not have so many spawn points. %d vehicles and %d pedestrains created." % (veh_num, 0))

        # ----- SPAWN VEHICLES AND PEDESTRIANS -------------------------------------------------------------------------

        # spawn own car
        my_car = world.spawn_actor(vehicle_blueprint[0], spawn_points[0])
        print("Created own car:", my_car.attributes)
        my_car.set_autopilot(1)
        actor_list.append(my_car)

        # spawn other vehicles
        # this can lead to another stupid error if spawn points are too close, thus the hacky try-except stuff below...
        for i in range(veh_num):
            try:
                other_vehicle = world.spawn_actor(random.choice(vehicle_blueprint), spawn_points[i + 1])
                print("Created vehicle", i + 1, "/", veh_num)
                other_vehicle.set_autopilot(1)
                actor_list.append(other_vehicle)
            except RuntimeError:
                if len(spawn_points) > veh_num + ped_num + 1:
                    veh_num += 1
                elif ped_num > 0:
                    ped_num -= 1
                    veh_num += 1

        # spawn pedestrians
        for i in range(ped_num):
            pedestrian = world.spawn_actor(random.choice(pedestrian_blueprint), spawn_points[i + veh_num + 1])
            pedestrian_controller = world.spawn_actor(pedestrian_controller_blueprint, carla.Transform(), pedestrian)
            print("Created pedestrian", i+1, "/", ped_num)
            actor_list.append(pedestrian)
            actor_list.append(pedestrian_controller)
            world.tick()
            pedestrian_controller.start()
            pedestrian_controller.go_to_location(world.get_random_location_from_navigation())
            pedestrian_controller.set_max_speed(1.4)

        # ----- DEFINE SENSORS AND ATTACH TO OWN CAR -------------------------------------------------------------------
        transform_front = carla.Transform(carla.Location(x=0.8, z=1.65))

        # semseg sensor
        semantic_front = world.spawn_actor(sensor_blueprint, transform_front, attach_to=my_car)
        print("Created semantic camera:", semantic_front.attributes)
        actor_list.append(semantic_front)

        # RGB sensor
        rgb_front = world.spawn_actor(rgb_blueprint, transform_front, attach_to=my_car)
        print("Created RGB camera:", rgb_front.attributes)
        actor_list.append(rgb_front)

        # ----- CREATE SYNC QUEUES FOR SENSOR DATA ---------------------------------------------------------------------
        semseg_queue = queue.Queue()
        frame_queue = queue.Queue()
        semantic_front.listen(semseg_queue.put)
        rgb_front.listen(frame_queue.put)

        # ----- RUN SIMULATION -----------------------------------------------------------------------------------------
        print("\n##### RUNNING SIMULATION #####")
        counter = -50  # wait 50 frames before starting to record to make sure we record actual driving

        while True:
            world.tick()
            counter += 1
            # world.wait_for_tick()  # maybe this is actually needed. We keep it here just in case we're not synced.

            # save current images
            if counter >= 0 and counter % time_inter == 0:

                image = semseg_queue.get()
                semantic_output_path = os.path.join(semseg_dir, "%06d.png" % image.frame_number)
                semantic_image_generator(image.raw_data, semantic_output_path, width=window_width, height=window_height)

                image = frame_queue.get()
                image.save_to_disk(os.path.join(frame_dir, "%06d.png" % image.frame_number))

            # empty queues
            elif counter % time_inter == (time_inter - 1):
                semseg_queue.queue.clear()
                frame_queue.queue.clear()

            print("Current frame:", counter)

            if counter >= stop_after:
                break

    # ----- CLEANUP ----------------------------------------------------------------------------------------------------
    finally:
        print("\n##### FINISHING SIMULATION #####")
        # client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        for actor in actor_list:
            actor_id = actor.id
            actor.destroy()
            print("Actor %d destroyed." % actor_id)

        print('Disabling synchronous mode...')
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        pygame.quit()
        print("\nSIMULATION COMPLETE.")


def build_arg_parser():
    """
    builds an argparser to run the application from command line
    :return: argparser
    """
    parser = argparse.ArgumentParser(
        description="Generate Carla videos with semantic segmentation maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_runs", "-n", type=int, default=1, help="How many simulations to run.")
    parser.add_argument("--fps", "-fps", type=int, default=20, help="At how many FPS the simulation is running.")
    parser.add_argument("--num_frames", "-f", type=int, default=200, help="How many frames to record.")
    parser.add_argument("--time_inter", "-t", type=int, default=1, help="Time interval between recorded frames.")
    parser.add_argument("--vehicles", "-v", type=int, default=80, help="Number of other vehicles.")
    parser.add_argument("--pedestrians", "-p", type=int, default=0, help="Number of pedestrians.")
    parser.add_argument("--output_dir", "-o", default="./images/", help="Data output directory.")
    parser.add_argument("--window_width", "-ww", type=int, default=512, help="Width of the Carla window.")
    parser.add_argument("--window_height", "-wh", type=int, default=256, help="Height of the Carla window.")
    parser.add_argument("--fov", "-fov", default=120, help="Horizontal field of view.")
    parser.add_argument("--sensor_tick", "-st", default=0.0, help="Number of seconds between sensor measurements.")
    return parser


def main():
    """Main function"""
    parser = build_arg_parser()
    args = parser.parse_args()
    name_with_time = "_".join("_".join(str(datetime.datetime.now()).split(" ")).split(":"))[:16]

    for i in range(args.num_runs):
        #towns = ["Town01", "Town02", "Town04", "Town05"]
        #town = random.choice(towns)
        town = "Town01" if i < args.num_runs * 1/3 else "Town02" if i < args.num_runs * 2/3 else "Town05"
        run_dir = os.path.join(os.curdir, "images", name_with_time, "%06d" % i + "_" + town)
        run_carla(
            fps=args.fps, time_inter=args.time_inter, num_frames=args.num_frames, other_vehicles=args.vehicles,
            pedestrians=args.pedestrians, window_width=args.window_width, window_height=args.window_height,
            fov=args.fov, sensor_tick=args.sensor_tick, output_dir=run_dir, town=town
        )


if __name__ == '__main__':
    main()
