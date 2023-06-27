import os
import queue
import subprocess
import time


def main():
    processes = []

    # GPUs available
    gpu_queue = queue.Queue()
    for i in [0, 1, 3]:
        gpu_queue.put(i)

    # Queue to hold scripts
    script_arg_queue = queue.Queue()
    for d in ["multiviewx", "wildtrack"]:
        for world_feat in ["conv", "deform_trans"]:
            for augmentation in ["0", "1"]:
                for alpha in ["0.0", "1.0"]:
                    script_arg_queue.put(
                        [
                            "main.py",
                            "-d",
                            d,
                            "--world_feat",
                            world_feat,
                            "--augmentation",
                            augmentation,
                            "--alpha",
                            alpha,
                        ]
                    )

    total = script_arg_queue.qsize()
    while not script_arg_queue.empty():
        if gpu_queue.empty():
            # No GPU is available, wait for a process to finish
            for process in processes:
                if (
                    process.poll() is not None
                ):  # A None value indicates that the process is still running
                    processes.remove(process)
                    gpu_queue.put(process.gpu_id)
                    break
            else:
                # No process has finished, wait a bit before checking again
                time.sleep(1.0)
                continue

        gpu_id = gpu_queue.get()
        script_arg = script_arg_queue.get()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        process = subprocess.Popen(["python"] + script_arg, env=env)
        process.gpu_id = gpu_id
        processes.append(process)
        print(f"[{total - script_arg_queue.qsize()} / {total}]")

    # Wait for all processes to finish
    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
