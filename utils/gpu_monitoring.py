import subprocess

def get_gpu_memory_info(info_type='used'):
    """
    Returns specific GPU memory information based on the info_type parameter.
    Valid options for info_type: 'used', 'free', 'total'
    """
    # Run nvidia-smi command to query memory info
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, encoding='utf-8'
    )

    # Parse the result
    lines = result.stdout.strip().split('\n')

    for line in lines:
        total, used, free = line.split(', ')
        if info_type == 'used':
            return int(used)  # Return used memory in MiB
        elif info_type == 'free':
            return int(free)  # Return free memory in MiB
        elif info_type == 'total':
            return int(total)  # Return total memory in MiB
        else:
            raise ValueError("Invalid info_type. Choose 'used', 'free', or 'total'.")

