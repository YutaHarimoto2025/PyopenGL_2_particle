# PyOpenGL Particle System - AI Coding Instructions

## Project Overview
This is a high-performance particle simulation and visualization application built with **Python**, **PyOpenGL**, and **PyQt6**. It supports hardware acceleration via **CuPy** (with NumPy fallback) and features a threaded physics simulation loop separate from the rendering loop.

## Architecture & Core Components

### 1. Application Structure
- **Entry Point:** `main_window.py` initializes the `MainWindow` and `GLWidget`.
- **UI (PyQt6):** `MainWindow` handles the GUI (toolbar, dock widgets), while `GLWidget` handles the OpenGL context and input events.
- **Rendering (PyOpenGL):**
  - `GLWidget.py`: Manages the render loop, camera, and scene setup.
  - `rendering.py`: Contains renderer classes (`ObjectRenderer`, `InstancedBallRendererColor`) and shader management.
  - `Shader/`: Contains GLSL vertex and fragment shaders.
- **Simulation:**
  - `simulation_buffer.py`: Runs the physics simulation in a separate thread (`_step_loop`).
  - `particle_system.py`: Handles particle logic.
  - `physics/`: Contains collision detection and other physics modules.

### 2. Data & Compute Abstraction (`tools.py`)
- **`xp` Abstraction:** The project dynamically selects **CuPy** (`cp`) if CUDA is available, otherwise falls back to **NumPy** (`np`).
  - **Rule:** Always use `tools.xp` for array operations to support both CPU and GPU.
  - **Rule:** Use `tools.to_numpy(array)` to safely convert data to a CPU NumPy array (e.g., for rendering or saving).
- **Parameters:**
  - `param.yaml`: Static configuration (loaded via `tools.param`).
  - `param_changable.json`: Dynamic runtime parameters (loaded via `tools.param_changable`).

### 3. Object Management
- **`Object3D` (`object3d.py`):** Base class for 3D objects, holding position, rotation, scale, and color.
- **Creation:** Use `create_obj.py` for generating geometries (boxes, axes, balls).

## Critical Workflows & Patterns

### Threading Model
- **Rendering Thread:** Main GUI thread (PyQt6). Accesses `simbuff.objects` for drawing.
- **Simulation Thread:** Background thread started by `simbuff.start_stepping()`. Updates physics state.
- **Synchronization:** Be mindful of race conditions when modifying shared state between `GLWidget` and `SimBuffer`.

### Rendering Pipeline
1. **Setup:** `initializeGL` in `GLWidget` sets up renderers and shaders.
2. **Update:** `update` calls `repaint()`.
3. **Draw:** `paintGL` clears the screen and calls `renderer.draw(obj)` for each object.
4. **Shaders:** Use `graphic_tools.load_shader` or `build_GLProgram`. Uniforms are cached in `ObjectRenderer.__init__`.

### Math & Geometry
- **GLM:** Use `glm` for vector and matrix math (e.g., `glm.vec3`, `glm.mat4`).
- **Coordinates:** Right-handed system (OpenGL standard).

## Development Guidelines

### Adding New Features
- **New Object Type:**
  1. Define geometry in `create_obj.py`.
  2. Add to `SimBuffer.objects` in `simulation_buffer.py`.
  3. Ensure `ObjectRenderer` handles its specific uniforms or create a new renderer in `rendering.py`.
- **New Shader:**
  1. Add `.vert` and `.frag` files to `Shader/`.
  2. Update `rendering.py` to load and use the new shader program.

### Common Tasks
- **Video Recording:** Handled by `MovieFFmpeg` in `movie_ffmpeg.py`. Controlled via `param.is_saving`.
- **Logging:** Object states are logged to JSONL files in `result/` if enabled.

### Code Style
- **Type Hinting:** Use Python type hints (`List`, `Dict`, `Optional`, etc.).
- **Imports:** Group imports: standard lib -> 3rd party (PyQt, OpenGL, glm) -> local modules.
