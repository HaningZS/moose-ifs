[Kernels]
  [./k1]
    type = Diffusion
    variable = T
  [../]
  [k2]
    type = TimeDerivative
    variable = T
  []
