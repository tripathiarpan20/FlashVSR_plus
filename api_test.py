import requests
import time
import argparse
import sys

def log(msg, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

def main():
    parser = argparse.ArgumentParser(description="Test FlashVSR+ API")
    parser.add_argument("--video-url", required=True, help="URL of the input video")
    parser.add_argument("--api-url", required=True, help="Base URL of the deployed API (e.g., http://localhost:9000)")
    args = parser.parse_args()

    base_url = args.api_url.rstrip('/')
    video_url = args.video_url

    # 1. Upload
    log(f"Initiating upload for video: {video_url}")
    upload_endpoint = f"{base_url}/api/v1/videos/upload"
    try:
        response = requests.post(upload_endpoint, json={"video_url": video_url})
        response.raise_for_status()
        upload_data = response.json()
        task_id = upload_data.get("task_id")
        log(f"Upload initiated. Task ID: {task_id}")
    except Exception as e:
        log(f"Upload failed: {e}", "ERROR")
        sys.exit(1)

    # 2. Wait for download to complete
    log("Waiting for video to be downloaded by the server...")
    status_endpoint = f"{base_url}/api/v1/videos/{task_id}/status"
    while True:
        try:
            response = requests.get(status_endpoint)
            response.raise_for_status()
            status_data = response.json()
            status = status_data.get("status")
            log(f"Current status: {status}")
            
            if status == "downloaded":
                log("Video downloaded successfully.")
                break
            elif status == "failed":
                log(f"Task failed during download: {status_data.get('message')}", "ERROR")
                sys.exit(1)
            
            time.sleep(2)
        except Exception as e:
            log(f"Status check failed: {e}", "ERROR")
            sys.exit(1)

    # 3. Start upscaling
    log(f"Starting upscaling for task: {task_id}")
    upscale_endpoint = f"{base_url}/api/v1/videos/{task_id}/upscaling"
    try:
        # Default parameters
        upscale_params = {
            "scale": 2,
            "mode": "tiny",
            "model_version": "v1.1"
        }
        response = requests.post(upscale_endpoint, json=upscale_params)
        response.raise_for_status()
        log("Upscaling request accepted.")
    except Exception as e:
        log(f"Upscaling request failed: {e}", "ERROR")
        sys.exit(1)

    # 4. Polling for completion
    log("Polling for processing completion...")
    while True:
        try:
            response = requests.get(status_endpoint)
            response.raise_for_status()
            status_data = response.json()
            status = status_data.get("status")
            message = status_data.get("message", "")
            
            log(f"Status: {status} | Message: {message}")
            
            if status == "completed":
                output_url = status_data.get("output_url")
                log(f"SUCCESS! Output URL: {output_url}", "FINISH")
                break
            elif status == "failed":
                log(f"Processing failed: {message}", "ERROR")
                sys.exit(1)
            elif status == "cancelled":
                log("Task was cancelled.", "WARNING")
                sys.exit(1)
                
            time.sleep(5)
        except Exception as e:
            log(f"Polling failed: {e}", "ERROR")
            sys.exit(1)

if __name__ == "__main__":
    main()
