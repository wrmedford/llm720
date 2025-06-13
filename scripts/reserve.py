#!/usr/bin/env python3
"""
Lambda Cloud Auto-Reserve Script
Interactive instance type selection with automatic reservation when available.
Plays a sound notification on macOS when successful.
"""

import os
import sys
import time
import json
import subprocess
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Configuration
API_BASE_URL = "https://cloud.lambda.ai"
DEFAULT_POLL_INTERVAL = 30  # Default polling interval in seconds

# Sound files on macOS
SUCCESS_SOUND = "/System/Library/Sounds/Glass.aiff"  # Success sound
ERROR_SOUND = "/System/Library/Sounds/Basso.aiff"    # Error sound
NOTIFICATION_SOUND = "/System/Library/Sounds/Hero.aiff"  # Found available


class LambdaCloudClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    def get_instance_types(self) -> Optional[Dict]:
        """Fetch available instance types."""
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/instance-types",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching instance types: {e}")
            return None
    
    def get_ssh_keys(self) -> Optional[List[Dict]]:
        """Fetch SSH keys to use for instance launch."""
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/ssh-keys",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching SSH keys: {e}")
            return None
    
    def launch_instance(self, region: str, instance_type: str, ssh_key_name: str, instance_name: str = None) -> Optional[Dict]:
        """Launch an instance in the specified region."""
        payload = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": [ssh_key_name]
        }
        
        if instance_name:
            payload["name"] = instance_name
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/instance-operations/launch",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error launching instance: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None


def play_sound(sound_file: str):
    """Play a sound file on macOS using afplay."""
    try:
        subprocess.run(["afplay", sound_file], check=True)
    except subprocess.CalledProcessError:
        print(f"Could not play sound: {sound_file}")


def display_instance_types(instance_types_data: Dict) -> List[Tuple[str, Dict]]:
    """Display instance types in a formatted table and return sorted list."""
    instance_list = []
    
    print("\n📊 Available Instance Types")
    print("=" * 120)
    print(f"{'#':<4} {'Type':<25} {'GPU Description':<30} {'GPUs':<5} {'vCPUs':<6} {'RAM':<8} {'$/hr':<8} {'Available'}")
    print("-" * 120)
    
    # Sort by price (highest to lowest) for better visibility of premium instances
    for type_name, type_data in sorted(
        instance_types_data.items(), 
        key=lambda x: x[1]['instance_type']['price_cents_per_hour'], 
        reverse=True
    ):
        instance = type_data['instance_type']
        specs = instance['specs']
        available_regions = type_data.get('regions_with_capacity_available', [])
        availability = f"✅ {len(available_regions)} regions" if available_regions else "❌ None"
        
        instance_list.append((type_name, type_data))
        idx = len(instance_list)
        
        price = instance['price_cents_per_hour'] / 100
        gpu_desc = instance.get('gpu_description', 'N/A')
        ram_gb = specs['memory_gib']
        
        print(f"{idx:<4} {type_name:<25} {gpu_desc:<30} {specs['gpus']:<5} {specs['vcpus']:<6} {ram_gb:<7}G ${price:<7.2f} {availability}")
    
    print("-" * 120)
    return instance_list


def select_instance_type(instance_list: List[Tuple[str, Dict]]) -> Optional[Tuple[str, Dict]]:
    """Let user select an instance type from the list."""
    while True:
        try:
            print("\n🎯 Enter the number of the instance type you want to monitor (or 'q' to quit):")
            choice = input("Your choice: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(instance_list):
                return instance_list[idx]
            else:
                print(f"❌ Please enter a number between 1 and {len(instance_list)}")
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")


def check_instance_availability(client: LambdaCloudClient, instance_type_name: str) -> Optional[Tuple[List[str], Dict]]:
    """Check if specific instance type is available in any region."""
    instance_types = client.get_instance_types()
    
    if not instance_types:
        return None
    
    # Check if instance type is in the response
    instance_data = instance_types.get("data", {}).get(instance_type_name)
    
    if not instance_data:
        return None
    
    # Check regions with capacity
    regions_available = instance_data.get("regions_with_capacity_available", [])
    
    if regions_available:
        # Return all available regions
        region_names = [r["name"] for r in regions_available]
        return region_names, instance_data
    
    return None


def format_instance_info(instance_data: Dict, region: str) -> str:
    """Format instance information for display."""
    instance = instance_data.get("instance_type", {})
    specs = instance.get("specs", {})
    
    info = f"""
🎉 Instance Available! 🎉
================================
Region: {region}
Type: {instance.get('description', 'N/A')}
GPU: {instance.get('gpu_description', 'N/A')}
vCPUs: {specs.get('vcpus', 'N/A')}
Memory: {specs.get('memory_gib', 'N/A')} GiB
Storage: {specs.get('storage_gib', 'N/A')} GiB
Price: ${instance.get('price_cents_per_hour', 0) / 100:.2f}/hour
================================
"""
    return info


def display_monitoring_info(instance_type_name: str, instance_data: Dict, auto_reserve: bool, poll_interval: int):
    """Display information about the instance type being monitored."""
    instance = instance_data['instance_type']
    specs = instance['specs']
    
    print(f"\n🔍 Monitoring Configuration")
    print("=" * 60)
    print(f"Instance Type: {instance_type_name}")
    print(f"Description: {instance['description']}")
    print(f"GPU: {instance['gpu_description']}")
    print(f"Specs: {specs['vcpus']} vCPUs, {specs['memory_gib']} GiB RAM, {specs['storage_gib']} GiB storage")
    print(f"Price: ${instance['price_cents_per_hour'] / 100:.2f}/hour")
    print(f"Poll Interval: {poll_interval} seconds")
    print(f"Auto-Reserve: {'✅ Enabled' if auto_reserve else '❌ Disabled (will ask for confirmation)'}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop monitoring\n")


def main():
    # Get API key from environment variable
    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        print("Error: LAMBDA_API_KEY environment variable not set")
        print("Please set it with: export LAMBDA_API_KEY='your-api-key'")
        sys.exit(1)
    
    client = LambdaCloudClient(api_key)
    
    # Get SSH keys
    print("🔑 Fetching SSH keys...")
    ssh_keys = client.get_ssh_keys()
    if not ssh_keys:
        print("Error: No SSH keys found. Please add an SSH key to your Lambda account.")
        sys.exit(1)
    
    # Use the first SSH key
    ssh_key_name = ssh_keys[0]["name"]
    print(f"✅ Using SSH key: {ssh_key_name}")
    
    # Fetch and display instance types
    print("\n📡 Fetching available instance types...")
    instance_types_response = client.get_instance_types()
    
    if not instance_types_response or "data" not in instance_types_response:
        print("❌ Error: Could not fetch instance types")
        sys.exit(1)
    
    instance_types_data = instance_types_response["data"]
    
    # Display instance types and let user select
    instance_list = display_instance_types(instance_types_data)
    
    # Show summary
    available_count = sum(1 for _, data in instance_list if data.get('regions_with_capacity_available', []))
    print(f"\n📈 Summary: {len(instance_list)} instance types total, {available_count} currently available")
    
    selected = select_instance_type(instance_list)
    if not selected:
        print("\n👋 Exiting...")
        sys.exit(0)
    
    instance_type_name, instance_data = selected
    
    # Ask about auto-reserve
    print("\n🤖 Enable auto-reserve? (y/n)")
    print("   y = Automatically reserve when available")
    print("   n = Ask for confirmation before reserving")
    auto_reserve = input("Your choice: ").strip().lower() == 'y'
    
    # Ask about polling interval
    print(f"\n⏱️  Set polling interval in seconds (default: {DEFAULT_POLL_INTERVAL}, minimum: 10):")
    poll_input = input("Poll interval (press Enter for default): ").strip()
    
    try:
        poll_interval = int(poll_input) if poll_input else DEFAULT_POLL_INTERVAL
        poll_interval = max(10, poll_interval)  # Minimum 10 seconds
    except ValueError:
        poll_interval = DEFAULT_POLL_INTERVAL
        print(f"Invalid input, using default: {DEFAULT_POLL_INTERVAL} seconds")
    
    # Display monitoring information
    display_monitoring_info(instance_type_name, instance_data, auto_reserve, poll_interval)
    
    poll_count = 0
    
    try:
        while True:
            poll_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"[{timestamp}] Poll #{poll_count}: Checking {instance_type_name} availability...", end="", flush=True)
            
            availability = check_instance_availability(client, instance_type_name)
            
            if availability:
                region, current_instance_data = availability
                print(f" ✅ FOUND in {region}!")
                
                # Play notification sound
                play_sound(NOTIFICATION_SOUND)
                
                # Display instance info
                print(format_instance_info(current_instance_data, region))
                
                # Check if we should auto-reserve or ask for confirmation
                should_launch = auto_reserve
                
                if not auto_reserve:
                    # Ask user if they want to launch
                    print("\n🚀 Do you want to launch this instance? (y/n): ", end="", flush=True)
                    confirm = input().strip().lower()
                    should_launch = (confirm == 'y')
                
                if should_launch:
                    # Attempt to launch the instance
                    print(f"{'Auto-launching' if auto_reserve else 'Attempting to launch'} instance in {region}...")
                    instance_name = f"{instance_type_name}-Auto-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    
                    result = client.launch_instance(region, instance_type_name, ssh_key_name, instance_name)
                    
                    if result:
                        instance_ids = result.get("data", {}).get("instance_ids", [])
                        if instance_ids:
                            print(f"\n🚀 SUCCESS! Instance launched with ID: {instance_ids[0]}")
                            print(f"Instance name: {instance_name}")
                            play_sound(SUCCESS_SOUND)
                            
                            # Show how to connect
                            print("\nTo get connection details, run:")
                            print(f"curl -H 'Authorization: Bearer $LAMBDA_API_KEY' {API_BASE_URL}/api/v1/instances/{instance_ids[0]}")
                            
                            break
                    else:
                        print("\n❌ Failed to launch instance!")
                        play_sound(ERROR_SOUND)
                        # Continue polling
                else:
                    print("Continuing to monitor...")
            else:
                print(" ❌ Not available")
            
            # Wait before next poll
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        play_sound(ERROR_SOUND)
        sys.exit(1)


if __name__ == "__main__":
    main()