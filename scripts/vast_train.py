#!/usr/bin/env python3
"""VAST.ai remote training helper.

Handles the full lifecycle: find instance → rent → upload → train → download → destroy.
Reports actual training cost at the end.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run(cmd, capture=True, check=True):
    """Run a shell command and return stdout."""
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True, check=check)
    return result.stdout.strip() if capture else None


def search_instances(min_gpu_ram=8, max_dph=1.0):
    """Search for available VAST instances, sorted by price."""
    query = (
        f"gpu_ram >= {min_gpu_ram} "
        f"reliability > 0.95 "
        f"num_gpus == 1 "
        f"cuda_vers >= 12.0 "
        f"dph_total <= {max_dph}"
    )
    raw = run(f"vastai search offers '{query}' --order 'dph' --limit 5 --raw")
    offers = json.loads(raw)

    print(f"\n{'ID':>8}  {'GPU':<18} {'VRAM':>6} {'$/hr':>8} {'DL Perf':>8}")
    print("-" * 55)
    for o in offers:
        print(f"{o['id']:>8}  {o['gpu_name']:<18} {o['gpu_ram']:>5.0f}G {o['dph_total']:>7.3f} {o.get('dlperf', 0):>8.1f}")

    return offers


def create_instance(offer_id, image="ultralytics/ultralytics:latest", disk_gb=20):
    """Rent a VAST instance."""
    print(f"\nCreating instance {offer_id} with image {image}...")
    result = run(f"vastai create instance {offer_id} --image {image} --disk {disk_gb} --raw")
    instance_info = json.loads(result)
    instance_id = instance_info.get("new_contract")
    print(f"Instance created: {instance_id}")
    return instance_id


def wait_for_instance(instance_id, timeout=300):
    """Wait for instance to be running and return SSH details."""
    print("Waiting for instance to start", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        raw = run(f"vastai show instance {instance_id} --raw")
        info = json.loads(raw)
        status = info.get("actual_status", "unknown")
        if status == "running":
            ssh_host = info.get("ssh_host")
            ssh_port = info.get("ssh_port")
            print(f"\nInstance running! SSH: ssh -p {ssh_port} root@{ssh_host}")
            return info
        print(".", end="", flush=True)
        time.sleep(10)
    raise TimeoutError(f"Instance {instance_id} did not start within {timeout}s")


def upload_dataset(ssh_host, ssh_port, local_dataset_dir, remote_dir="/workspace/dataset"):
    """Upload dataset to VAST instance via SCP."""
    print(f"\nUploading dataset from {local_dataset_dir}...")
    run(f"scp -P {ssh_port} -r {local_dataset_dir} root@{ssh_host}:{remote_dir}", capture=False)
    print("Dataset uploaded.")


def upload_file(ssh_host, ssh_port, local_path, remote_path):
    """Upload a single file via SCP."""
    run(f"scp -P {ssh_port} {local_path} root@{ssh_host}:{remote_path}", capture=False)


def run_remote(ssh_host, ssh_port, command):
    """Run a command on the VAST instance via SSH."""
    return run(f"ssh -p {ssh_port} root@{ssh_host} '{command}'", capture=True, check=True)


def download_weights(ssh_host, ssh_port, remote_path, local_path):
    """Download trained weights from VAST instance."""
    print(f"\nDownloading weights to {local_path}...")
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    run(f"scp -P {ssh_port} root@{ssh_host}:{remote_path} {local_path}", capture=False)
    print(f"Weights saved to: {local_path}")


def destroy_instance(instance_id):
    """Destroy a VAST instance to stop billing."""
    print(f"\nDestroying instance {instance_id}...")
    run(f"vastai destroy instance {instance_id}")
    print("Instance destroyed. Billing stopped.")


def main():
    parser = argparse.ArgumentParser(description="VAST.ai remote training")
    sub = parser.add_subparsers(dest="command")

    # Search
    search_p = sub.add_parser("search", help="Search for available GPUs")
    search_p.add_argument("--min-vram", type=int, default=8, help="Minimum GPU VRAM in GB")
    search_p.add_argument("--max-cost", type=float, default=1.0, help="Max $/hr")

    # Train (full lifecycle)
    train_p = sub.add_parser("train", help="Full remote training lifecycle")
    train_p.add_argument("--offer-id", type=int, required=True, help="VAST offer ID to rent")
    train_p.add_argument("--dataset", type=str, required=True, help="Local dataset directory")
    train_p.add_argument("--train-script", type=str, default="scripts/train.py",
                         help="Local training script to upload")
    train_p.add_argument("--train-args", type=str, default="",
                         help="Extra args to pass to training script")
    train_p.add_argument("--output", type=str, default="outputs/train/vast-run/weights",
                         help="Local directory for downloaded weights")

    # Destroy
    destroy_p = sub.add_parser("destroy", help="Destroy a running instance")
    destroy_p.add_argument("--instance-id", type=int, required=True)

    args = parser.parse_args()

    if args.command == "search":
        search_instances(args.min_vram, args.max_cost)

    elif args.command == "train":
        # Full lifecycle
        start_time = datetime.now()

        # 1. Rent instance
        instance_id = create_instance(args.offer_id)
        try:
            # 2. Wait for startup
            info = wait_for_instance(instance_id)
            ssh_host = info["ssh_host"]
            ssh_port = info["ssh_port"]
            dph = info.get("dph_total", 0)

            # 3. Upload dataset + training script
            upload_dataset(ssh_host, ssh_port, args.dataset)
            upload_file(ssh_host, ssh_port, args.train_script, "/workspace/train.py")

            # 4. Run training
            print("\n=== Starting training ===")
            remote_data = "/workspace/dataset/data.yaml"
            train_cmd = (
                f"cd /workspace && python3 train.py "
                f"--data {remote_data} "
                f"--project /workspace/outputs --name run "
                f"{args.train_args}"
            )
            print(f"Remote command: {train_cmd}")
            run_remote(ssh_host, ssh_port, train_cmd)

            # 5. Download weights
            download_weights(
                ssh_host, ssh_port,
                "/workspace/outputs/run/weights/best.pt",
                f"{args.output}/best.pt"
            )
            download_weights(
                ssh_host, ssh_port,
                "/workspace/outputs/run/weights/last.pt",
                f"{args.output}/last.pt"
            )

            # 6. Report cost
            end_time = datetime.now()
            duration_hrs = (end_time - start_time).total_seconds() / 3600
            total_cost = duration_hrs * dph

            print(f"\n{'=' * 50}")
            print(f"TRAINING COST REPORT")
            print(f"{'=' * 50}")
            print(f"  GPU:        {info.get('gpu_name', 'unknown')}")
            print(f"  Rate:       ${dph:.3f}/hr")
            print(f"  Duration:   {duration_hrs:.2f} hrs")
            print(f"  Total cost: ${total_cost:.2f}")
            print(f"{'=' * 50}")

            # Save cost report
            cost_report = {
                "gpu": info.get("gpu_name"),
                "rate_per_hour": dph,
                "duration_hours": round(duration_hrs, 2),
                "total_cost": round(total_cost, 2),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "instance_id": instance_id,
            }
            report_path = Path(args.output) / "cost_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(cost_report, f, indent=2)
            print(f"Cost report saved to: {report_path}")

        finally:
            # 7. ALWAYS destroy instance
            destroy_instance(instance_id)

    elif args.command == "destroy":
        destroy_instance(args.instance_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
