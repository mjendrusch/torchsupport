import torch
import torch.nn as nn
import torch.nn.functional as func

class GeometricVectorPerceptron(nn.Module):
  r"""Implements the Geometric Vector Perceptron layer (GVP) from
  arXiv:2009.01411 (Jing et al. 2020). It transforms a set of scalar
  and vector features in a rotation equivariant way.

  Args:
    in_scalars (int): number of input scalar features.
    in_vectors (int): number of input vector features.
    out_scalars (int): number of output scalar features.
    out_vectors (int): number of output vector features.
    hidden_vectors (int): number of hidden vector features.
    scalar_activation (callable): activation applied to scalar features.
    vector_activation (callable): activation applied to scale vector features.

  Shape:
    - Scalar inputs: :math:`(N, C_{S, in})`
    - Vector inputs: :math:`(N, 3, C_{V, in})`
    - Scalar outputs: :math:`(N, C_{S, out})`
    - Vector outputs: :math:`(N, 3, C_{V, out})`
  """
  def __init__(self, in_scalars, in_vectors,
               out_scalars, out_vectors,
               hidden_vectors=None,
               scalar_activation=func.relu,
               vector_activation=torch.sigmoid):
    super().__init__()
    hidden_vectors = hidden_vectors or max(in_vectors, out_vectors)
    self.scalar_activation = scalar_activation
    self.vector_activation = vector_activation
    self.project_vectors = nn.Linear(in_vectors, hidden_vectors, bias=False)
    self.predict_vectors = nn.Linear(hidden_vectors, out_vectors, bias=False)
    self.project_scalars = nn.Linear(in_scalars + hidden_vectors, out_scalars)

  def forward(self, scalars, vectors):
    vector_projection = self.project_vectors(vectors)
    vector_prediction = self.predict_vectors(vector_projection)
    projection_norm = vector_projection.norm(p=2, dim=1)
    prediction_norm = vector_prediction.norm(p=2, dim=1, keepdim=True)
    scalar_projection = self.project_scalars(
      torch.cat((scalars, projection_norm), dim=1)
    )
    scalar_result = self.scalar_activation(scalar_projection)
    vector_result = self.vector_activation(prediction_norm) * vector_prediction
    return scalar_result, vector_result
