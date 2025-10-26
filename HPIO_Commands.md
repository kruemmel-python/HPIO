# 🧠 HPIO – Optimierungs- & Videoaufnahme-Kommandos

Diese Übersicht listet **alle sinnvollen Startbefehle** für den HPIO‑Recorder (Version mit Video‑Aufzeichnung und Presets).  
Alle Befehle sind **einzeilig**, sofort in PowerShell oder CMD ausführbar.

---

## 🎯 Ackley (GPU)

### 1️⃣ Standard GPU‑Modus
```powershell
python hpio_record.py ackley --gpu --video runs/ackley_gpu.mkv --fps 24 --size 1600x900 --overlay --report-every 5
```

### 2️⃣ Tight‑Variante (schnelle Konvergenz, weiches Feld)
```powershell
python hpio_record.py ackley --gpu --ackley-tight --video runs/ackley_gpu_tight.mkv --fps 24 --size 1600x900 --overlay --report-every 5
```

### 3️⃣ Pro‑Preset (optimierte Schärfe und Phasenführung)
```powershell
python hpio_record.py ackley --gpu --preset ackley-gpu-pro --video runs/ackley_gpu_pro.mkv --fps 24 --size 1600x900 --overlay --report-every 5
```

### 4️⃣ Aggressives Ackley‑Profil (Flag‑Variante)
```powershell
python hpio_record.py ackley --gpu --ackley-pro --video runs/ackley_gpu_aggr.mkv --fps 24 --size 1600x900 --overlay --report-every 5
```

---

## 💡 Rastrigin (CPU)

### 1️⃣ Standard CPU‑Modus
```powershell
python hpio_record.py rastrigin --video runs/rastrigin_std.mp4 --fps 30 --size 1280x720 --overlay --report-every 5
```

### 2️⃣ CPU‑Pro‑Profil (bewährt, stabil, präzise)
```powershell
python hpio_record.py rastrigin --cpu-pro --video runs/rastrigin_cpu_pro.mp4 --fps 30 --size 1280x720 --overlay --report-every 5
```

### 3️⃣ CPU‑Preset‑Profil (optimiert gegen 1.0‑Ring‑Lock)
```powershell
python hpio_record.py rastrigin --preset rastrigin-cpu-pro --video runs/rastrigin_cpu_preset.mp4 --fps 30 --size 1280x720 --overlay --report-every 5
```

---

## 🧩 Himmelblau (GPU oder CPU)

### 1️⃣ Standard GPU‑Modus
```powershell
python hpio_record.py himmelblau --gpu --video runs/himmelblau_gpu.mkv --fps 30 --size 1600x900 --overlay --report-every 5
```

### 2️⃣ Standard CPU‑Modus
```powershell
python hpio_record.py himmelblau --video runs/himmelblau_cpu.mp4 --fps 30 --size 1280x720 --overlay --report-every 5
```

---

## ⚙️ Debug‑ und Benchmark‑Runs

### 1️⃣ Minimale Darstellung (ohne Overlay, weniger Frames)
```powershell
python hpio_record.py rastrigin --preset rastrigin-cpu-pro --video runs/rastrigin_fast.mp4 --fps 15 --size 1024x576 --viz-freq 3
```

### 2️⃣ GPU‑Vergleich mit unterschiedlichen Seeds
```powershell
python hpio_record.py ackley --gpu --preset ackley-gpu-pro --seed 42 --video runs/ackley_gpu_seed42.mkv --fps 24 --size 1600x900 --overlay --report-every 5
python hpio_record.py ackley --gpu --preset ackley-gpu-pro --seed 99 --video runs/ackley_gpu_seed99.mkv --fps 24 --size 1600x900 --overlay --report-every 5
```

---

## 📁 Ergebnisdateien

Alle erzeugten Videos werden automatisch in den jeweiligen **`runs/`**‑Unterordner geschrieben.  
Die Dateiformate `.mp4` und `.mkv` sind voll kompatibel mit allen modernen Playern.

---

© 2025 – HPIO Optimization Suite  
Autor: Ralf Krümmel · Konzept: Hybrid Phase Interaction Optimization (HPIO)
