import torch

def read_obj_to_tensor(filename, num_vertices=5023):
  """
  Reads an OBJ file and returns a torch tensor of vertices.

  Args:
      filename: Path to the OBJ file.
      num_vertices: Expected number of vertices in the file (optional).

  Returns:
      A torch tensor of shape (1, num_vertices, 3) containing vertex coordinates.
  """
  vertices = []

  with open(filename, 'r') as f:
    for line in f:
      parts = line.strip().split()
      if parts[0] == 'v':
        # Vertex definition (v x y z)
        vertices.append([float(p) for p in parts[1:]])

  # Check if the number of vertices matches expectations (optional)
  if len(vertices) != num_vertices:
    print(f"Warning: Expected {num_vertices} vertices, found {len(vertices)}")

  # Convert list to a torch tensor with desired shape
  vertices_tensor = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0)

  return vertices_tensor

# Example usage
filename = '/mnt/HDD3/nguyen/04_flame_based/rome/samples/meshes/meshes_yixuan_exp_img/00004.obj'
num_vertices = 5023  # Adjust if you know the expected number of vertices
vertices_tensor = read_obj_to_tensor(filename, num_vertices)

print("Vertices tensor shape:", vertices_tensor.shape)

# You can now use the vertices_tensor for your downstream tasks
