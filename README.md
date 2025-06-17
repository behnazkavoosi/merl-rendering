# MERL BRDF Rendering with Mitsuba 3

This repository implements a rendering pipeline using **measured BRDFs** (e.g., from the MERL dataset) in **Mitsuba 3** with a custom Python-based BSDF and CUDA acceleration. 

---

## ✨ Features

- 📊 **Measured BRDF Support** — Loads BRDFs as 4D tensors with angular mapping.
- ⚙️ **Custom Mitsuba 3 BSDF** — Implements `MyBSDF` using PyTorch & Dr.Jit.
- ⚡ **GPU-Accelerated** — Uses Mitsuba 3’s `cuda_ad_rgb` variant for fast rendering.
- 🛠️ **Scripted Rendering** — Batch render scenes with different materials and views.

---

## 🚀 Getting Started

### 1. Install Mitsuba 3

Follow the [official Mitsuba 3 build guide](https://mitsuba.readthedocs.io/en/latest/src/getting_started/compiling.html)

### 2. Install Python Dependencies

```bash
pip install torch drjit tqdm scipy
```

## 🧪 Usage

### Render a Scene with a Measured BRDF

To render a scene using a specific measured BRDF:

```bash
python render_measured.py <material_name> <scene_file.xml>
```



