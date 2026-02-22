import cupy as cp
import numpy as np

class ParticleSystem:
    def __init__(self, num_particles=10000, bounds=(-5, 5)):
        self.num_particles = num_particles
        self.bounds = bounds

        # Initialize positions randomly within bounds
        self.positions = cp.random.uniform(bounds[0], bounds[1], (num_particles, 3)).astype(cp.float32)
        
        # Initialize velocities randomly
        self.velocities = cp.random.uniform(-1, 1, (num_particles, 3)).astype(cp.float32)
        
        # Initialize colors (random)
        self.colors = cp.random.uniform(0, 1, (num_particles, 3)).astype(cp.float32)
        
        # Initialize radii
        self.radii = cp.full((num_particles,), 0.05, dtype=cp.float32)

        # Pre-allocate model matrices array (N, 4, 4)
        # We use a transposed structure so that row-major memory layout matches OpenGL column-major matrix
        self.model_matrices = cp.zeros((num_particles, 4, 4), dtype=cp.float32)
        
        # Set identity-like structure (diagonal)
        # We will update the diagonal (scale) and the last row (translation) each frame
        # [ s, 0, 0, 0 ]
        # [ 0, s, 0, 0 ]
        # [ 0, 0, s, 0 ]
        # [ x, y, z, 1 ]  <-- This row becomes the last column in OpenGL
        
        # Set w component of the last column (which is [3, 3] here) to 1.0
        self.model_matrices[:, 3, 3] = 1.0

    def update(self, dt):
        # Update positions: p = p + v * dt
        self.positions += self.velocities * dt

        # Simple boundary collision (bounce)
        # Check lower bounds
        hit_lower = self.positions < self.bounds[0]
        self.positions[hit_lower] = self.bounds[0]
        self.velocities[hit_lower] *= -1

        # Check upper bounds
        hit_upper = self.positions > self.bounds[1]
        self.positions[hit_upper] = self.bounds[1]
        self.velocities[hit_upper] *= -1

    def get_render_data(self):
        """
        Returns (model_matrices, colors) as numpy arrays for OpenGL.
        """
        # Update model matrices
        # Reset scale (in case radii changed, though they are constant here)
        # We can optimize this by only setting it once if static, but let's be safe
        
        # Set scale (diagonal elements 0,0; 1,1; 2,2)
        # Using advanced indexing to set diagonal for all matrices
        # shape (N,)
        r = self.radii
        self.model_matrices[:, 0, 0] = r
        self.model_matrices[:, 1, 1] = r
        self.model_matrices[:, 2, 2] = r
        
        # Set translation (last row in our transposed structure -> last column in GL)
        self.model_matrices[:, 3, :3] = self.positions

        # Transfer to CPU (NumPy)
        # Note: For very high performance with mapped buffers, we could avoid this copy,
        # but for 10^5 particles, this copy is acceptable (~6MB data).
        
        # We return the arrays. The renderer expects (N, 4, 4) and (N, 3).
        return cp.asnumpy(self.model_matrices), cp.asnumpy(self.colors)

    def set_radii(self, radius):
        self.radii[:] = radius
