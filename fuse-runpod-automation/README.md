# RunPod LLaVA Automation - Complete Setup Guide

This package automates the entire process of processing 3000+ artwork images with LLaVA on RunPod GPU.

## What This Does

1. ✅ Uploads your 3000 images to RunPod (one-time, ~30-60 min)
2. ✅ Runs LLaVA processing on GPU (~20-30 hours)
3. ✅ Downloads results back to your computer
4. ✅ Handles checkpoints (resume if interrupted)
5. ✅ You can close your computer - it runs on RunPod

## Files Included

- `runpod_llava_processor.py` - Main processing script (runs on RunPod)
- `automate_runpod.sh` - Automation script (runs on your computer)
- `README.md` - This file

## Prerequisites

- RunPod account (runpod.io)
- SSH key configured on your computer
- Images folder with your 3000 artwork images

## Setup Instructions

### Step 1: Create RunPod Pod

1. Go to https://runpod.io
2. Click **"Pods"** → **"GPU Cloud"**
3. Select **"PyTorch"** template
4. Choose GPU:
   - **Recommended**: RTX 4090 (~$0.44/hour) or RTX A4000 (~$0.34/hour)
   - **Budget**: RTX 3090 (~$0.29/hour) - slower but cheaper
5. Set **Container Disk**: 50GB minimum
6. Click **"Deploy On-Demand"**
7. Wait for pod to start (~30 seconds)

### Step 2: Get SSH Connection Details

1. On your running pod, click **"Connect"**
2. Select **"Connect via SSH"**
3. You'll see something like:
   ```
   ssh root@ssh.runpod.io -p 12345 -i ~/.ssh/id_rsa
   ```
4. Note down:
   - Host: `ssh.runpod.io`
   - Port: `12345`

### Step 3: Configure Automation Script

1. Open `automate_runpod.sh` in a text editor
2. Edit these lines:
   ```bash
   LOCAL_IMAGES_DIR="./images"              # Path to your 3k images
   LOCAL_OUTPUT_DIR="./results"             # Where to save results
   RUNPOD_SSH_HOST="ssh.runpod.io"         # From step 2
   RUNPOD_SSH_PORT="12345"                  # From step 2
   ```
3. Save the file

### Step 4: Run the Automation

Open terminal and run:

```bash
./automate_runpod.sh
```

The script will:
1. Upload images (30-60 min) - Progress bar shown
2. Upload processing script
3. Start processing (20-30 hours for 3k images)
4. Download results when done

**Important**: You can close the terminal during upload/processing - it continues on RunPod!

## Manual Alternative (If Automation Fails)

If the bash script doesn't work, you can do it manually:

### Upload Images Manually

```bash
# Replace with your SSH details
rsync -avz --progress \
  -e "ssh -p 12345 -i ~/.ssh/id_rsa" \
  ./images/ \
  root@ssh.runpod.io:/workspace/images/
```

### Upload Script

```bash
scp -P 12345 -i ~/.ssh/id_rsa \
  runpod_llava_processor.py \
  root@ssh.runpod.io:/workspace/
```

### Connect and Run

```bash
ssh -p 12345 -i ~/.ssh/id_rsa root@ssh.runpod.io

# Once connected:
cd /workspace
python3 runpod_llava_processor.py
```

### Download Results

```bash
scp -P 12345 -i ~/.ssh/id_rsa \
  root@ssh.runpod.io:/workspace/output/llava_captions.jsonl \
  ./results/
```

## Monitoring Progress

### Option 1: SSH into RunPod

```bash
ssh -p 12345 -i ~/.ssh/id_rsa root@ssh.runpod.io

# Check output file
tail -f /workspace/output/llava_captions.jsonl

# Count processed images
wc -l /workspace/output/llava_captions.jsonl
```

### Option 2: Use RunPod Web Terminal

1. Go to your pod on RunPod website
2. Click **"Connect"** → **"Start Web Terminal"**
3. Run:
   ```bash
   tail -f /workspace/output/llava_captions.jsonl
   ```

## Resuming After Interruption

The script has checkpoint support! If processing stops:

1. Just run the automation script again
2. It will skip already-processed images
3. Continue from where it left off

## Cost Estimate

For 3000 images:

- **RTX 4090** (~30s/image): 25 hours × $0.44/hr = **~$11**
- **RTX A4000** (~40s/image): 33 hours × $0.34/hr = **~$11**
- **RTX 3090** (~45s/image): 37 hours × $0.29/hr = **~$11**

All options cost about the same - choose RTX 4090 for speed!

## Troubleshooting

### "Permission denied" error
```bash
chmod +x automate_runpod.sh
```

### "rsync: command not found"
Install rsync:
- **Mac**: `brew install rsync`
- **Linux**: `sudo apt-get install rsync`
- **Windows**: Use WSL or manual upload method

### "Connection refused"
- Check your pod is running on RunPod
- Verify SSH host and port are correct
- Make sure your SSH key is added to ssh-agent

### Images not uploading
Check your internet connection and try manual upload method above

### Want to pause and resume later?
Just press Ctrl+C - the checkpoint system saves progress every 10 images!

## After Processing Completes

1. **Download your results** (script does this automatically)
2. **Stop your RunPod pod** to avoid charges!
   - Go to RunPod website
   - Click "Stop" on your pod
3. Your results are in: `./results/llava_captions.jsonl`

## Next Steps

Once you have `llava_captions.jsonl`, you can:

1. Clean and structure the JSON (use the cleaning script from earlier)
2. Proceed with Step 1: Vectorization
3. Step 2: Web search for artwork metadata
4. Step 3-4: Similarity matching

## Support

If you run into issues:
- Check RunPod pod is running
- Verify SSH connection works manually
- Check RunPod balance (need credits)
- Make sure images folder exists and has images

Good luck! 🚀
