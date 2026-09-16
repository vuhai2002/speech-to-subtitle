# 🚀 Vertex AI Setup Guide for Batch Transcribe

A guide to configuring **Vertex AI (Service Account)** for step 1 of the pipeline: `transcribe/batch_transcribe_vertex.py` (MP3 -> TXT). Costs go to Google Cloud billing, so the **$300 Free Trial Credit** can be used.

> **⚠️ IMPORTANT NOTE ABOUT MODELS:**
> At the time of writing (03/2026), the **Gemini 3.x preview** models (gemini-3.1-pro-preview, gemini-3-pro-preview, etc.) show up in the model list but **do NOT work over the API** on a Free Trial project (404 error). Only the following models are confirmed to work:
> - ✅ `gemini-2.5-flash` — **Recommended** (fast, cheap, good quality)
> - ✅ `gemini-2.5-pro` — Higher quality, slower, more prone to rate limits
> - ✅ `gemini-2.5-flash-lite` — Cheapest and fastest

---

## Table of Contents

1. [Enable APIs](#step-1-enable-apis-in-the-google-cloud-console)
2. [Create a Service Account](#step-2-create-a-service-account)
3. [Create a JSON Key](#step-3-create-and-download-a-json-key)
4. [Create a GCS Bucket](#step-4-create-a-google-cloud-storage-bucket)
5. [Configure .env](#step-5-configure-the-env-file)
6. [Install Dependencies](#step-6-install-dependencies)
7. [Run the Script](#step-7-run-the-script)
8. [Troubleshooting](#troubleshooting)

---

## Step 1: Enable APIs in the Google Cloud Console

Go to the [Google Cloud Console](https://console.cloud.google.com) and select your project.

### 1a. Enable the Vertex AI API
1. Open [APIs & Services -> Library](https://console.cloud.google.com/apis/library)
2. Search for **"Vertex AI API"**
3. Click **Enable**

### 1b. Enable the Cloud Storage API
1. Search for **"Cloud Storage API"** (or **"Google Cloud Storage JSON API"**)
2. Click **Enable** (it may already be enabled)

---

## Step 2: Create a Service Account

1. Open [IAM & Admin -> Service Accounts](https://console.cloud.google.com/iam-admin/service-accounts)
2. Click **"+ CREATE SERVICE ACCOUNT"** (at the top)
3. Fill in the details:
   - **Service account name:** `transcribe-worker`
   - **Service account ID:** auto-generated (e.g. `transcribe-worker`)
   - **Description:** `Service account for batch transcribe`
4. Click **"CREATE AND CONTINUE"**
5. **Assign roles** (important!):
   - Click **"+ ADD ANOTHER ROLE"** to add each role:
     - **`Vertex AI User`** -> allows calling Gemini via Vertex AI
     - **`Storage Object Admin`** -> allows uploading/deleting files on GCS
   - Click **"CONTINUE"**
6. Click **"DONE"**

> **📝 Note:** For a quick test, you can assign the **Owner** role to the service account. In production, however, use least privilege (Vertex AI User + Storage Object Admin).

---

## Step 3: Create and Download a JSON Key

1. In the Service Accounts list, click **transcribe-worker** (the one you just created)
2. Select the **"KEYS"** tab
3. Click **"ADD KEY"** -> **"Create new key"**
4. Select **JSON** -> click **"CREATE"**
5. The JSON file downloads automatically (e.g. `<PROJECT_ID>-xxxx.json`)

### Upload the key file to the VPS

```bash
# From your local machine, use scp to upload to the VPS:
scp ~/Downloads/<PROJECT_ID>-xxxx.json <user>@<VPS_IP>:~/video-transcribe/service-account-key.json

# Or, if you're already on the VPS, copy the JSON file contents and paste them:
nano ~/video-transcribe/service-account-key.json
# Paste the JSON contents -> Ctrl+O -> Enter -> Ctrl+X
```

> **🔒 SECURITY:** Never commit the key file to Git! The name `service-account-key.json` is already in the repo's `.gitignore`; if you use a different name, add it to `.gitignore` yourself.

---

## Step 4: Create a Google Cloud Storage Bucket

The bucket is used to temporarily upload audio files before sending them to Gemini for processing. The script **automatically deletes** the file on GCS once transcription is done.

### Option 1: Via the Console (UI)
1. Open [Cloud Storage -> Buckets](https://console.cloud.google.com/storage/browser)
2. Click **"+ CREATE"**
3. Fill in:
   - **Bucket name:** `transcribe-audio-<PROJECT_ID>` (must be globally unique)
   - **Location type:** Region
   - **Region:** `us-central1` (same region as Vertex AI)
   - **Storage class:** Standard
   - **Access control:** Uniform (default)
4. Click **"CREATE"**

### Option 2: Via the gcloud CLI
```bash
gcloud storage buckets create gs://transcribe-audio-<PROJECT_ID> \
    --project=<PROJECT_ID> \
    --location=us-central1 \
    --uniform-bucket-level-access
```

---

## Step 5: Configure the `.env` File

Open the `.env` file in the project directory and add/edit the following lines:

```env
# Model used for transcription
# Recommended: gemini-2.5-flash (fast, cheap) or gemini-2.5-pro (high quality)
GEMINI_MODEL=gemini-2.5-flash

# === VERTEX AI CONFIG ===
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
GCS_BUCKET_NAME=your_bucket_name
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### Explanation of the variables:

| Variable | Description | Example |
|---|---|---|
| `GEMINI_MODEL` | Gemini model to use | `gemini-2.5-flash` |
| `GOOGLE_CLOUD_PROJECT` | Project ID on GCP | `<PROJECT_ID>` |
| `GOOGLE_CLOUD_LOCATION` | Region for Vertex AI | `us-central1` |
| `GCS_BUCKET_NAME` | GCS bucket name | `transcribe-audio-<PROJECT_ID>` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to the JSON key file | `~/video-transcribe/service-account-key.json` |

---

## Step 6: Install Dependencies

```bash
cd ~/video-transcribe
pip install google-genai google-cloud-storage google-auth python-dotenv
```

### Required packages:

| Package | Purpose |
|---|---|
| `google-genai` | SDK for calling Gemini via Vertex AI |
| `google-cloud-storage` | Upload/delete files on GCS |
| `google-auth` | Authenticate with the Service Account |
| `python-dotenv` | Read configuration from the `.env` file |

---

## Step 7: Run the Script

```bash
cd ~/video-transcribe

# Run with the default configuration (5 workers)
python3 transcribe/batch_transcribe_vertex.py

# Adjust the number of workers (lower it if you hit rate limits often)
python3 transcribe/batch_transcribe_vertex.py --workers 3

# Specify input/output directories
python3 transcribe/batch_transcribe_vertex.py --mp3-dir /path/to/mp3 --txt-dir /path/to/txt

# Re-run everything (including files that already have TXT)
python3 transcribe/batch_transcribe_vertex.py --force
```

### Notes when running:
- The script automatically **skips** files that already have TXT (resume-friendly)
- The script reads the `.env` next to the script first; if there isn't one, it reads the `.env` in the parent directory (repo root). If you scp only the script file onto a VM, put the `.env` in the same directory as the script.
- A log file is created next to the script: `transcribe/batch_transcribe_vertex_YYYYMMDD_HHMMSS.log` (gitignored)
- The output TXT is the input to step 2 (`python -m realign.run_batch --txt-dir ...`), see `README.md`
- On a **rate limit** (429), the script automatically retries with increasing backoff
- On **499 CANCELLED** or **503 UNAVAILABLE**, the script retries automatically

### Model recommendations:

| Model | Speed | Cost | Rate Limit | Good for |
|---|---|---|---|---|
| `gemini-2.5-flash` | ⚡ Fast (~60s/file) | 💰 Cheap | Rare | Large batches, saving credit |
| `gemini-2.5-pro` | 🐢 Slow (~80-120s/file) | 💰💰 More expensive | Frequent | When high quality is needed |

---

## Troubleshooting

### ❌ "Permission denied" when uploading to GCS
-> Check that the Service Account has the **Storage Object Admin** role.
Go to [IAM & Admin -> IAM](https://console.cloud.google.com/iam-admin/iam) -> find the service account -> add the role.

### ❌ "Vertex AI API has not been enabled"
-> Go back to [Step 1a](#1a-enable-the-vertex-ai-api) and enable the Vertex AI API.

### ❌ "Could not automatically determine credentials"
-> Check that `GOOGLE_APPLICATION_CREDENTIALS` in `.env` points to the correct path of the JSON key file.

### ❌ "Bucket not found"
-> Check that `GCS_BUCKET_NAME` in `.env` matches the name of the bucket you created.

### ❌ "404 NOT_FOUND" for a model
-> The model is not available on your project. Change `GEMINI_MODEL` to `gemini-2.5-flash`.

Check which models work:
```bash
# Model test script
python3 -c "
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.oauth2 import service_account

load_dotenv('.env')
creds = service_account.Credentials.from_service_account_file(
    os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)
client = genai.Client(vertexai=True, project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                      location='us-central1', credentials=creds)

for m in ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-lite']:
    try:
        r = client.models.generate_content(model=m, contents='Hello',
            config=types.GenerateContentConfig(max_output_tokens=20))
        print(f'✅ {m}: OK')
    except Exception as e:
        print(f'❌ {m}: {str(e)[:60]}')
"
```

### ❌ "429 RESOURCE_EXHAUSTED" (Rate Limit)
-> This is normal when running many workers. The script retries automatically. If it happens too often, reduce the number of workers:
```bash
python3 transcribe/batch_transcribe_vertex.py --workers 3
```

### ❌ "499 CANCELLED"
-> Server timeout when processing a large file. The script retries automatically. If it still fails, try running again.

### ❌ "FAILED_PRECONDITION" (on the very first run)
-> Google is setting up Vertex AI service agents for the project. Wait 2-3 minutes and run again.
