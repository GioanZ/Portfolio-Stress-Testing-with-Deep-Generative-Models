import os
import tensorflow as tf

GPU_MEMORY_LIMITS = {1: 1024, 2: 2048, 3: 3072, 4: 4096, 6: 6144, 8: 8192, 12: 12288}


def gpu_available(use_gpu=True, gpu_gb_use=8):
    if not use_gpu:
        # Disable GPU usage completely
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("GPU disabled. Running on CPU only.")
        return

    gpus = tf.config.list_physical_devices("GPU")
    print("Number of GPUs Available:", len(gpus))

    mem_lim_gpu = GPU_MEMORY_LIMITS.get(gpu_gb_use, None)
    if mem_lim_gpu is None:
        raise ValueError(
            f"Invalid GPU_GB_USE value: {gpu_gb_use}. Choose from {list(GPU_MEMORY_LIMITS.keys())}."
        )

    if gpus:
        print("Using GPU:", gpus)
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=mem_lim_gpu)],
            )
            print(f"GPU memory limited to {gpu_gb_use}GB ({mem_lim_gpu}MB)")
        except RuntimeError as e:
            print("Error configuring GPU:", e)
    else:
        print("No GPU found. Running on CPU.")
