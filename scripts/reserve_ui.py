#!/usr/bin/env python3
"""
Lambda Cloud Auto-Reserve Terminal UI
Interactive terminal interface for monitoring and auto-reserving Lambda Cloud instances.
"""

import os
import sys
import time
import json
import subprocess
import requests
import curses
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Configuration
API_BASE_URL = "https://cloud.lambda.ai"
DEFAULT_POLL_INTERVAL = 30  # Default polling interval in seconds

# Sound files on macOS
SUCCESS_SOUND = "/System/Library/Sounds/Glass.aiff"
ERROR_SOUND = "/System/Library/Sounds/Basso.aiff"
NOTIFICATION_SOUND = "/System/Library/Sounds/Hero.aiff"


class AppState(Enum):
    SETUP = "setup"
    INSTANCE_LIST = "instance_list"
    MONITORING = "monitoring"
    REGION_SELECT = "region_select"


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
    
    def launch_instance(self, region: str, instance_type: str, ssh_key_name: str, instance_name: str = None) -> Tuple[Optional[Dict], Optional[str]]:
        """Launch an instance in the specified region. Returns (result, error_message)."""
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
            return response.json(), None
        except requests.exceptions.HTTPError as e:
            # Extract detailed error info
            error_detail = f"HTTP {e.response.status_code}"
            try:
                error_json = e.response.json()
                if "error" in error_json:
                    error_detail = f"{error_json['error'].get('message', 'Unknown error')}"
            except:
                error_detail = e.response.text[:200]  # First 200 chars of response
            
            return None, error_detail
        except requests.exceptions.RequestException as e:
            return None, f"Request error: {str(e)}"


def play_sound(sound_file: str):
    """Play a sound file on macOS using afplay."""
    try:
        subprocess.run(["afplay", sound_file], check=True, capture_output=True)
    except:
        pass  # Silently fail if sound can't be played


class TerminalUI:
    def __init__(self, client: LambdaCloudClient, ssh_key_name: str):
        self.client = client
        self.ssh_key_name = ssh_key_name
        self.state = AppState.SETUP
        self.instance_list = []
        self.selected_index = 0
        self.scroll_offset = 0
        self.filter_available_only = False
        self.auto_reserve = False
        self.poll_interval = DEFAULT_POLL_INTERVAL
        self.monitoring_instance = None
        self.poll_count = 0
        self.last_poll_time = None
        self.status_messages = []
        self.available_regions = []
        self.selected_region_index = 0
        self.setup_stage = 0  # 0: auto-reserve, 1: poll interval
        
    def add_status_message(self, message: str, is_error: bool = False):
        """Add a status message to the display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_messages.append({
            'time': timestamp,
            'message': message,
            'is_error': is_error
        })
        # Keep only last 10 messages
        self.status_messages = self.status_messages[-10:]
    
    def draw_header(self, stdscr, title: str):
        """Draw the header section."""
        height, width = stdscr.getmaxyx()
        
        # Title
        header = f"╔{'═' * (width-2)}╗"
        stdscr.addstr(0, 0, header, curses.color_pair(4))
        
        title_text = f"Lambda Cloud Auto-Reserve - {title}"
        padding = (width - len(title_text) - 2) // 2
        stdscr.addstr(1, 0, f"║{' ' * padding}{title_text}{' ' * (width - padding - len(title_text) - 2)}║", curses.color_pair(4))
        
        footer = f"╚{'═' * (width-2)}╝"
        stdscr.addstr(2, 0, footer, curses.color_pair(4))
    
    def draw_instance_list(self, stdscr):
        """Draw the instance type selection screen."""
        height, width = stdscr.getmaxyx()
        
        self.draw_header(stdscr, "Select Instance Type")
        
        # Help text
        help_text = "↑/↓: Navigate | Enter: Select | a: Toggle available only | r: Refresh | c: Settings | q: Quit"
        stdscr.addstr(4, 2, help_text, curses.color_pair(3))
        
        # Column headers
        headers = f"{'Type':<25} {'GPU':<25} {'vCPUs':<6} {'RAM':<8} {'$/hr':<8} {'Status'}"
        stdscr.addstr(6, 2, headers, curses.color_pair(2) | curses.A_BOLD)
        stdscr.addstr(7, 0, "─" * width)
        
        # Filter display
        display_list = self.instance_list
        if self.filter_available_only:
            display_list = [(name, data) for name, data in self.instance_list 
                          if data.get('regions_with_capacity_available', [])]
        
        # Calculate visible area
        list_start_y = 8
        list_height = height - list_start_y - 5  # Leave room for status
        
        # Adjust scroll offset
        if self.selected_index < self.scroll_offset:
            self.scroll_offset = self.selected_index
        elif self.selected_index >= self.scroll_offset + list_height:
            self.scroll_offset = self.selected_index - list_height + 1
        
        # Draw instances
        for i in range(self.scroll_offset, min(len(display_list), self.scroll_offset + list_height)):
            y = list_start_y + (i - self.scroll_offset)
            name, data = display_list[i]
            instance = data['instance_type']
            specs = instance['specs']
            regions = data.get('regions_with_capacity_available', [])
            
            # Format fields
            gpu_desc = instance.get('gpu_description', 'N/A')[:24]
            price = f"${instance['price_cents_per_hour'] / 100:.2f}"
            ram = f"{specs['memory_gib']}G"
            status = f"✓ {len(regions)} regions" if regions else "✗ Unavailable"
            
            # Highlight selected row
            attr = curses.A_REVERSE if i == self.selected_index else 0
            if regions:
                attr |= curses.color_pair(1)  # Green for available
            
            row = f" {name:<24} {gpu_desc:<25} {specs['vcpus']:<6} {ram:<8} {price:<8} {status}"
            stdscr.addstr(y, 0, row[:width-1], attr)
        
        # Status bar
        status_y = height - 3
        stdscr.addstr(status_y, 0, "─" * width)
        
        total = len(self.instance_list)
        available = sum(1 for _, data in self.instance_list if data.get('regions_with_capacity_available', []))
        filter_status = " (filtered)" if self.filter_available_only else ""
        status = f"Total: {total} | Available: {available} | Selected: {self.selected_index + 1}/{len(display_list)}{filter_status}"
        stdscr.addstr(status_y + 1, 2, status, curses.color_pair(3))
    
    def draw_monitoring(self, stdscr):
        """Draw the monitoring screen."""
        height, width = stdscr.getmaxyx()
        
        self.draw_header(stdscr, "Monitoring")
        
        if not self.monitoring_instance:
            return
        
        name, data = self.monitoring_instance
        instance = data['instance_type']
        specs = instance['specs']
        
        # Instance info
        y = 4
        stdscr.addstr(y, 2, "Instance Details", curses.color_pair(2) | curses.A_BOLD)
        y += 2
        
        info_lines = [
            f"Type: {name}",
            f"GPU: {instance.get('gpu_description', 'N/A')}",
            f"Specs: {specs['vcpus']} vCPUs, {specs['memory_gib']}G RAM, {specs['storage_gib']}G storage",
            f"Price: ${instance['price_cents_per_hour'] / 100:.2f}/hour",
        ]
        
        for line in info_lines:
            stdscr.addstr(y, 4, line)
            y += 1
        
        # Monitoring settings
        y += 1
        stdscr.addstr(y, 2, "Settings", curses.color_pair(2) | curses.A_BOLD)
        y += 2
        
        settings_lines = [
            f"Auto-Reserve: {'✓ Enabled' if self.auto_reserve else '✗ Manual confirmation'}",
            f"Poll Interval: {self.poll_interval} seconds",
            f"Polls Completed: {self.poll_count}",
        ]
        
        for line in settings_lines:
            stdscr.addstr(y, 4, line)
            y += 1
        
        # Current status
        y += 1
        stdscr.addstr(y, 2, "Status", curses.color_pair(2) | curses.A_BOLD)
        y += 2
        
        if self.last_poll_time:
            time_since = int(time.time() - self.last_poll_time)
            next_poll = max(0, self.poll_interval - time_since)
            status = f"Next poll in: {next_poll}s"
            
            # Check current availability
            availability = self.check_availability()
            if availability:
                regions, _ = availability
                stdscr.addstr(y, 4, f"🎉 AVAILABLE in {len(regions)} region(s)!", curses.color_pair(1) | curses.A_BOLD)
            else:
                stdscr.addstr(y, 4, status)
        else:
            stdscr.addstr(y, 4, "Starting...")
        
        # Status messages
        y = height - 12
        stdscr.addstr(y, 2, "Recent Activity", curses.color_pair(2) | curses.A_BOLD)
        y += 1
        stdscr.addstr(y, 0, "─" * width)
        y += 1
        
        for msg in self.status_messages[-8:]:  # Show last 8 messages
            color = curses.color_pair(5) if msg['is_error'] else curses.color_pair(3)
            stdscr.addstr(y, 2, f"[{msg['time']}] {msg['message']}"[:width-3], color)
            y += 1
        
        # Help
        help_text = "s: Stop monitoring | c: Change settings | q: Quit"
        stdscr.addstr(height - 2, 2, help_text, curses.color_pair(3))
    
    def draw_setup(self, stdscr):
        """Draw the setup screen."""
        height, width = stdscr.getmaxyx()
        
        self.draw_header(stdscr, "Initial Setup")
        
        # Center box
        box_height = 12
        box_width = 60
        box_y = (height - box_height) // 2
        box_x = (width - box_width) // 2
        
        # Draw setup box
        for i in range(box_height):
            stdscr.addstr(box_y + i, box_x, " " * box_width, curses.color_pair(4))
        
        if self.setup_stage == 0:
            # Auto-reserve setup
            title = "Auto-Reserve Configuration"
            stdscr.addstr(box_y + 2, box_x + (box_width - len(title)) // 2, title, 
                         curses.color_pair(2) | curses.A_BOLD)
            
            options = [
                ("Yes", "Automatically reserve when available"),
                ("No", "Ask for confirmation before reserving")
            ]
            
            for i, (option, desc) in enumerate(options):
                y = box_y + 5 + i * 2
                is_selected = (i == 0 and self.auto_reserve) or (i == 1 and not self.auto_reserve)
                attr = curses.A_REVERSE if is_selected else 0
                
                option_text = f"[{option}] {desc}"
                stdscr.addstr(y, box_x + 5, option_text, curses.color_pair(4) | attr)
            
            help_text = "↑/↓: Select | Enter: Continue | q: Quit"
            
        else:
            # Poll interval setup
            title = "Polling Interval"
            stdscr.addstr(box_y + 2, box_x + (box_width - len(title)) // 2, title, 
                         curses.color_pair(2) | curses.A_BOLD)
            
            intervals = [10, 20, 30, 60, 120]
            current_text = f"Current: {self.poll_interval} seconds"
            stdscr.addstr(box_y + 4, box_x + (box_width - len(current_text)) // 2, current_text)
            
            for i, interval in enumerate(intervals):
                y = box_y + 6 + i
                is_selected = interval == self.poll_interval
                attr = curses.A_REVERSE if is_selected else 0
                
                text = f"{interval} seconds"
                if interval == DEFAULT_POLL_INTERVAL:
                    text += " (default)"
                
                stdscr.addstr(y, box_x + (box_width - len(text)) // 2, text, 
                             curses.color_pair(4) | attr)
            
            help_text = "↑/↓: Select | Enter: Start Monitoring | q: Quit"
        
        # Help text
        help_x = box_x + (box_width - len(help_text)) // 2
        stdscr.addstr(box_y + box_height - 2, help_x, help_text, curses.color_pair(3))
    
    def handle_setup_input(self, key):
        """Handle input in setup screen."""
        if key == ord('q'):
            return False  # Quit
            
        if self.setup_stage == 0:
            # Auto-reserve selection
            if key in [curses.KEY_UP, curses.KEY_DOWN]:
                self.auto_reserve = not self.auto_reserve
            elif key == ord('\n'):  # Enter
                self.setup_stage = 1
                
        else:
            # Poll interval selection
            intervals = [10, 20, 30, 60, 120]
            current_idx = intervals.index(self.poll_interval) if self.poll_interval in intervals else 2
            
            if key == curses.KEY_UP and current_idx > 0:
                self.poll_interval = intervals[current_idx - 1]
            elif key == curses.KEY_DOWN and current_idx < len(intervals) - 1:
                self.poll_interval = intervals[current_idx + 1]
            elif key == ord('\n'):  # Enter
                self.state = AppState.INSTANCE_LIST
                self.add_status_message(f"Setup complete. Auto-reserve: {'Yes' if self.auto_reserve else 'No'}, Poll: {self.poll_interval}s")
        
        return True
    
    def check_availability(self):
        """Check if the monitored instance is available."""
        if not self.monitoring_instance:
            return None
        
        name, _ = self.monitoring_instance
        instance_types = self.client.get_instance_types()
        
        if not instance_types:
            return None
        
        instance_data = instance_types.get("data", {}).get(name)
        if not instance_data:
            return None
        
        regions = instance_data.get("regions_with_capacity_available", [])
        if regions:
            return [r["name"] for r in regions], instance_data
        
        return None
    
    def handle_instance_list_input(self, key):
        """Handle input in instance list view."""
        display_list = self.instance_list
        if self.filter_available_only:
            display_list = [(name, data) for name, data in self.instance_list 
                          if data.get('regions_with_capacity_available', [])]
        
        if key == curses.KEY_UP and self.selected_index > 0:
            self.selected_index -= 1
        elif key == curses.KEY_DOWN and self.selected_index < len(display_list) - 1:
            self.selected_index += 1
        elif key == ord('a'):
            self.filter_available_only = not self.filter_available_only
            self.selected_index = 0
            self.scroll_offset = 0
        elif key == ord('r'):
            self.refresh_instance_list()
        elif key == ord('c'):
            self.setup_stage = 0  # Reset to first setup screen
            self.state = AppState.SETUP
        elif key == ord('\n'):  # Enter
            if display_list:
                self.monitoring_instance = display_list[self.selected_index]
                self.state = AppState.MONITORING
                self.poll_count = 0
                self.last_poll_time = None
                self.add_status_message(f"Started monitoring {self.monitoring_instance[0]}")
        elif key == ord('q'):
            return False
        
        return True
    
    def handle_monitoring_input(self, key):
        """Handle input in monitoring view."""
        if key == ord('s'):
            self.state = AppState.INSTANCE_LIST
            self.add_status_message("Stopped monitoring")
        elif key == ord('c'):
            self.setup_stage = 0  # Reset to first setup screen
            self.state = AppState.SETUP
            self.add_status_message("Entering setup...")
        elif key == ord('q'):
            return False
        
        return True
    
    def handle_region_select_input(self, key):
        """Handle input in region selection."""
        if key == curses.KEY_UP and self.selected_region_index > 0:
            self.selected_region_index -= 1
        elif key == curses.KEY_DOWN and self.selected_region_index < len(self.available_regions) - 1:
            self.selected_region_index += 1
        elif key == ord('\n'):  # Enter
            return self.available_regions[self.selected_region_index]
        elif key == 27:  # Escape
            return None
        
        return True
    
    def refresh_instance_list(self):
        """Refresh the instance type list."""
        self.add_status_message("Refreshing instance list...")
        response = self.client.get_instance_types()
        
        if response and "data" in response:
            self.instance_list = sorted(
                response["data"].items(),
                key=lambda x: x[1]['instance_type']['price_cents_per_hour'],
                reverse=True
            )
            self.add_status_message("Instance list refreshed")
        else:
            self.add_status_message("Failed to refresh instance list", is_error=True)
    
    def monitoring_tick(self, stdscr):
        """Perform monitoring check. Returns True if app should exit."""
        if not self.monitoring_instance or self.state != AppState.MONITORING:
            return False
        
        current_time = time.time()
        if self.last_poll_time and (current_time - self.last_poll_time) < self.poll_interval:
            return False
        
        self.last_poll_time = current_time
        self.poll_count += 1
        
        name, _ = self.monitoring_instance
        self.add_status_message(f"Checking availability for {name}...")
        
        availability = self.check_availability()
        if availability:
            regions, instance_data = availability
            # Debug: log the type and content of regions
            self.add_status_message(f"Found in regions: {regions} (type: {type(regions).__name__})")
            self.add_status_message(f"🎉 {name} available in {len(regions)} region(s)!")
            play_sound(NOTIFICATION_SOUND)
            
            if self.auto_reserve:
                # Auto-reserve in first available region - ensure it's a string
                selected_region = regions[0] if isinstance(regions, list) else regions
                self.add_status_message(f"Auto-reserving in {selected_region}...")
                if self.launch_instance(selected_region):
                    return True  # Exit the app on success
            else:
                # Show region selection
                self.available_regions = regions if isinstance(regions, list) else [regions]
                self.selected_region_index = 0
                self.state = AppState.REGION_SELECT
        else:
            self.add_status_message(f"{name} not available")
        
        return False
    
    def launch_instance(self, region: str):
        """Attempt to launch an instance."""
        name, _ = self.monitoring_instance
        instance_name = f"{name}-Auto-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Ensure region is a string, not a list
        if isinstance(region, list):
            region = region[0]
            self.add_status_message(f"Warning: Region was a list, using first: {region}")
        
        self.add_status_message(f"Launching {name} in {region}...")
        api_response, error_msg = self.client.launch_instance(region, name, self.ssh_key_name, instance_name)
        
        if result and result.get("data", {}).get("instance_ids"):
            instance_id = result["data"]["instance_ids"][0]
            self.add_status_message(f"🚀 Success! Instance {instance_id} launched!")
            play_sound(SUCCESS_SOUND)
            
            # Show connection info
            self.add_status_message(f"Instance name: {instance_name}")
            self.add_status_message("Check status with: curl -H 'Authorization: Bearer $LAMBDA_API_KEY'")
            self.add_status_message(f"  {API_BASE_URL}/api/v1/instances/{instance_id}")
            return True
        else:
            # Check if the client logged an error
            self.add_status_message(f"Failed to launch instance in {region}", is_error=True)
            play_sound(ERROR_SOUND)
            return False
    
    def run(self, stdscr):
        """Main UI loop."""
        # Setup colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)
        
        # Setup
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(100)  # 100ms refresh rate
        
        # Main loop
        running = True
        while running:
            stdscr.clear()
            
            # Draw current screen
            if self.state == AppState.SETUP:
                self.draw_setup(stdscr)
            elif self.state == AppState.INSTANCE_LIST:
                # Load data if needed
                if not self.instance_list:
                    self.refresh_instance_list()
                self.draw_instance_list(stdscr)
            elif self.state == AppState.MONITORING:
                self.draw_monitoring(stdscr)
                # Check if monitoring tick wants us to exit (successful auto-launch)
                if self.monitoring_tick(stdscr):
                    running = False
            elif self.state == AppState.REGION_SELECT:
                self.draw_monitoring(stdscr)  # Keep monitoring screen as background
                self.draw_region_select(stdscr)
            
            stdscr.refresh()
            
            # Handle input
            try:
                key = stdscr.getch()
                if key != -1:  # -1 means no input
                    if self.state == AppState.SETUP:
                        running = self.handle_setup_input(key)
                    elif self.state == AppState.INSTANCE_LIST:
                        running = self.handle_instance_list_input(key)
                    elif self.state == AppState.MONITORING:
                        running = self.handle_monitoring_input(key)
                    elif self.state == AppState.REGION_SELECT:
                        result = self.handle_region_select_input(key)
                        if result is True:
                            continue
                        elif result is None:  # Cancelled
                            self.state = AppState.MONITORING
                        else:  # Region selected
                            if self.launch_instance(result):
                                running = False  # Exit on successful launch
                            else:
                                self.state = AppState.MONITORING
            except KeyboardInterrupt:
                running = False


def main():
    # Check for API key
    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        print("Error: LAMBDA_API_KEY environment variable not set")
        print("Please set it with: export LAMBDA_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Initialize client
    client = LambdaCloudClient(api_key)
    
    # Get SSH keys
    print("Fetching SSH keys...")
    ssh_keys = client.get_ssh_keys()
    if not ssh_keys:
        print("Error: No SSH keys found. Please add an SSH key to your Lambda account.")
        sys.exit(1)
    
    ssh_key_name = ssh_keys[0]["name"]
    print(f"Using SSH key: {ssh_key_name}")
    print("\nStarting terminal UI...")
    time.sleep(1)
    
    # Run the UI
    ui = TerminalUI(client, ssh_key_name)
    curses.wrapper(ui.run)
    
    print("\nThank you for using Lambda Cloud Auto-Reserve!")


if __name__ == "__main__":
    main()