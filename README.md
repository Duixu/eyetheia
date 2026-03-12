# EyeTheia

## Project Overview

**EyeTheia** is an open-source project dedicated to **2D gaze estimation**, predicting the user's **point of regard on the screen** in pixel coordinates.
It leverages pre-trained deep learning models to infer the gaze position from facial images and landmarks extracted in real time.

A **personal calibration phase** is performed for each user to collect sample data across multiple screen points.
These samples are then used to **fine-tune the model**, adapting it to the user's specific facial features and improving gaze prediction accuracy.

---

# Client-side ONNX Inference

EyeTheia now supports **client-side inference using ONNX** after the personal calibration and fine-tuning phase.

This allows the trained model to run **directly inside the browser** using **ONNX Runtime Web**, eliminating the need to send frames to the backend for gaze prediction once calibration is complete.

The workflow is as follows:

1. The user performs the **personal calibration phase**.
2. The FastAPI backend collects calibration samples.
3. The model is **fine-tuned on the server** using these samples.
4. The fine-tuned PyTorch model is **exported to ONNX**.
5. The browser **downloads the ONNX model, metadata, and normalization statistics**.
6. Gaze prediction is then performed **locally in the browser using WebWorkers and ONNX Runtime Web**.

This architecture provides several benefits:

* **Reduced latency** (no network round-trip per frame)
* **Lower backend load**
* **Better scalability for experiments with many participants**
* **Offline inference capability after calibration**

The frontend implementation relies on:

* **ONNX Runtime Web** for model execution
* **WebWorkers** for asynchronous inference
* **MediaPipe FaceMesh** for real-time facial landmark extraction

After the ONNX model is loaded, the system automatically switches to **frontend inference mode**.

---

# Environment Setup

We use **Conda** to manage the development environment.
To create the environment, run:

```bash
$ conda create -n eyetheia python=3.10
$ conda activate eyetheia
```

To install all dependencies:

```bash
$ make lib
```

---

# Running EyeTheia

You have two main ways to run **EyeTheia**, depending on your use case.

---

## 1. Run the full demo (tracking + calibration)

If you want to directly try the complete demo — including the calibration phase and real-time gaze tracking — simply run:

```bash
$ make run
```

This command launches the end-to-end application locally, handling camera input, calibration, and live gaze prediction.

---

# Start a Tracker Server (API Mode)

If you prefer to use EyeTheia as a backend service via its FastAPI interface, you can start a tracker server manually or through the Makefile.

Supported models:

* **baseline** — iTracker trained on the **GazeCapture** dataset
* **mpiiface** — iTracker retrained on the **MPIIFaceGaze** dataset

You can start them directly using the **Makefile** commands:

```bash
# Launch the baseline tracker (iTracker trained on GazeCapture)
$ make baseline
```

```bash
# Launch the MPIIFaceGaze retrained tracker
$ make mpii
```

Or manually:

```bash
python src/run_server.py --model_path MODEL_PATH [--host HOST]
```

Arguments:

* `--model_path` **(required)**: path to the model weights.
* `--host` **(optional)**: default `127.0.0.1`.

The port is automatically assigned depending on the model:

* **8001** → baseline
* **8002** → mpiiface

Each tracker runs its own **FastAPI server**, allowing **multiple models to run simultaneously on different ports**.

This is particularly useful for experiments comparing different gaze estimation models.

---

# ONNX Export API

After calibration and fine-tuning, the backend automatically exports the trained model to ONNX.

The following API endpoints are used by the frontend:

| Endpoint                     | Description                         |
| ---------------------------- | ----------------------------------- |
| `/onnx/export/{client_id}`   | Export the fine-tuned model to ONNX |
| `/onnx/status/{client_id}`   | Check export status                 |
| `/onnx/metadata/{client_id}` | Retrieve model metadata             |
| `/onnx/means/{client_id}`    | Retrieve normalization statistics   |
| `/onnx/latest/{client_id}`   | Download the ONNX model             |

Each model is stored using the pair:

```
(client_id, model_key)
```

Where:

* **client_id** identifies the experiment subject or session
* **model_key** identifies the backbone used for fine-tuning (`baseline` or `mpiiface`)

This allows multiple models for the same subject to coexist without conflicts.

Example:

```
subject_001 + baseline
subject_001 + mpiiface
```

---

# API Usage Example

A **JavaScript frontend example** is provided which:

* captures webcam frames
* extracts facial landmarks with MediaPipe
* performs calibration
* downloads the ONNX model
* runs gaze prediction in the browser

Example implementation: [pygaze.js – Calypso frontend example](https://git.interactions-team.fr/INTERACTIONS/calypso/src/branch/main/lib/web/survey/trackers/pygaze.js)


This implementation can serve as a reference for integrating **EyeTheia** into:

* behavioral experiments
* psychology studies
* human-computer interaction research
* gaze-controlled interfaces

---

# Documentation

We use **Sphinx** to generate project documentation.

To build the documentation:

```bash
make doc
```

The generated HTML files will be available in:

```
docs/_build/html/
```

The main entry point is:

```
docs/_build/html/index.html
```

---

# Testing

To run unit tests:

```bash
$ make test
```

---

# Dataset

The project supports two pre-trained model configurations:

### itracker_baseline.tar

Based on the original **iTracker** architecture from the paper:

"Eye Tracking for Everyone"

[https://arxiv.org/abs/1606.05814](https://arxiv.org/abs/1606.05814)

Trained on the **GazeCapture dataset**.

---

### itracker_mpiiface.tar

Retrained using the **MPIIFaceGaze dataset**:

[http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip](http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip)

This dataset contains real-world face images with accurate gaze annotations.

If you plan to train the model, download and extract the dataset into:

```
dataset/
```

For **inference only**, the dataset is **not required**.

---

# VM User Webcam

If you run EyeTheia inside a **virtual machine** (for example **WSL2**) and do not use the FastAPI webcam routes, please refer to the documentation section:

```
Running on a Virtual Machine (e.g., WSL2)
```

This section explains how to stream a webcam using **MJPEG Streamer** and configure the environment variable:

```
WEBCAM_URL
```

---

# License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

You may redistribute and/or modify it under the terms of the GPL-3.0 as published by the Free Software Foundation.

See the `LICENSE` file for full details.
