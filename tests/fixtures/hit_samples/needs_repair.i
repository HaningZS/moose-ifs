[Mesh]
  type=GeneratedMesh
  dim=2
[]

[Kernels]
  [./k1]
    type=Diffusion
    variable=T
  [../]
